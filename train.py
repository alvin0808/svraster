#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import json
import time
import uuid
import imageio
import datetime
import numpy as np
from tqdm import tqdm
from random import randint

import torch

from src.config import cfg, update_argparser, update_config

from src.utils.system_utils import seed_everything
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.bounding_utils import decide_main_bounding
from src.utils import mono_utils
from src.utils import loss_utils
from src.utils import octree_utils
from src.dataloader.data_pack import DataPack, compute_iter_idx
from src.sparse_voxel_model import SparseVoxelModel

import svraster_cuda


def training(args): 
    # Init and load data pack
    data_pack = DataPack(cfg.data, cfg.model.white_background) #data_pack.py -> DataPack

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    tr_cam_indices = compute_iter_idx(len(tr_cams), cfg.procedure.n_iter) #data_pack.py -> compute_iter_idx 학습 iter수만큼 indices 생성

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init() 

    # Prepare monocular depth priors if instructed
    if cfg.regularizer.lambda_depthanythingv2:
        mono_utils.prepare_depthanythingv2(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            force_rerun=False)

    if cfg.regularizer.lambda_mast3r_metric_depth:
        mono_utils.prepare_mast3r_metric_depth(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            mast3r_repo_path=cfg.regularizer.mast3r_repo_path)

    # Decide main (inside) region bounding box
    bounding = decide_main_bounding(
        cfg_bounding=cfg.bounding,
        tr_cams=tr_cams,
        pcd=data_pack.point_cloud,  # Not used
        suggested_bounding=data_pack.suggested_bounding,  # Can be None
    ) #  inner bounding box의 center와 radius를 결정하는 함수 5/7 
    

    # Init voxel model
    voxel_model = SparseVoxelModel(cfg.model) #본격적 으로 SparseVoxelModel을 생성하는 부분 
    
    if args.load_iteration: #중간저장된거 로드하는 경우
        loaded_iter = voxel_model.load_iteration(args.load_iteration)
    else:
        loaded_iter = None
        voxel_model.model_init( #from constructor.py 
            bounding=bounding,
            cfg_init=cfg.init,
            cfg_mode= cfg.model.density_mode,
            cameras=tr_cams)  

    first_iter = loaded_iter if loaded_iter else 1
    print(f"Start optmization from iters={first_iter}.") 

    # Init optimizer 옵티마이저 초기화화
    voxel_model.optimizer_init(cfg.optimizer)
    if loaded_iter and args.load_optimizer:
        voxel_model.optimizer_load_iteration(loaded_iter)

    # Init lr warmup scheduler, gradient gradualy increase
    # from 0 to base_lr in n_warmup iterations
    if first_iter <= cfg.optimizer.n_warmup:
        rate = max(first_iter - 1, 0) / cfg.optimizer.n_warmup
        for param_group in voxel_model.optimizer.param_groups:
            param_group["base_lr"] = param_group["lr"]
            param_group["lr"] = rate * param_group["base_lr"]

    # Init subdiv
    remain_subdiv_times = sum(
        (i >= first_iter)
        for i in range(
            cfg.procedure.subdivide_from, cfg.procedure.subdivide_until+1,
            cfg.procedure.subdivide_every
        ) #range(1000, 15001, 1000)
    )
    subdivide_scale = cfg.procedure.subdivide_target_scale ** (1 / remain_subdiv_times)
    subdivide_prop = max(0, (subdivide_scale - 1) / 7)
    print(f"Subdiv: times={remain_subdiv_times:2d} scale-each-time={subdivide_scale*100:.1f}% prop={subdivide_prop*100:.1f}%")
    #Subdiv: times=15 scale-each-time=135.0% prop=5.0% 라고 뜨는거 보니
    # 15번 subdivide를 해야하고, 매번 135%씩 scale을 늘려야 한다는 뜻인듯 #5/10
    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True) #execute time 측정 in GPU
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0 #elapsed time

    tr_render_opt = {
        'track_max_w': False, # track max_w for subdivision
        'lambda_R_concen': cfg.regularizer.lambda_R_concen, # lambda_R_concen loss 
        'output_T': False, # ray marching 중 누적 투과율 T
        'output_depth': False, #true 이면 depth를 출력
        'ss': 1.0,  # disable supersampling at first super-sampling ratio
        'rand_bg': cfg.regularizer.rand_bg, # random background color
        'use_auto_exposure': cfg.auto_exposure.enable, # use auto exposure
    }
    #조건부 활성화를 위해 아래와 같이 정의
    sparse_depth_loss = loss_utils.SparseDepthLoss(
        iter_end=cfg.regularizer.sparse_depth_until)
    depthanythingv2_loss = loss_utils.DepthAnythingv2Loss(
        iter_from=cfg.regularizer.depthanythingv2_from,
        iter_end=cfg.regularizer.depthanythingv2_end,
        end_mult=cfg.regularizer.depthanythingv2_end_mult)
    mast3r_metric_depth_loss = loss_utils.Mast3rMetricDepthLoss(
        iter_from=cfg.regularizer.mast3r_metric_depth_from,
        iter_end=cfg.regularizer.mast3r_metric_depth_end,
        end_mult=cfg.regularizer.mast3r_metric_depth_end_mult)
    nd_loss = loss_utils.NormalDepthConsistencyLoss(
        iter_from=cfg.regularizer.n_dmean_from,
        iter_end=cfg.regularizer.n_dmean_end,
        ks=cfg.regularizer.n_dmean_ks,
        tol_deg=cfg.regularizer.n_dmean_tol_deg)
    nmed_loss = loss_utils.NormalMedianConsistencyLoss(
        iter_from=cfg.regularizer.n_dmed_from,
        iter_end=cfg.regularizer.n_dmed_end)

    ema_loss_for_log = 0.0 #use for logging
    ema_psnr_for_log = 0.0 #use for logging
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training")   #10 iterations마다 processbar update
    #parameters for eikonal loss
    vox_size_min_inv = 1.0 / voxel_model.vox_size.min().item() #복셀의 최소 크기
    print(f"voxel_model.vox_size_min_inv = {vox_size_min_inv:.4f}") #복셀의 최소 크기
    max_voxel_level = voxel_model.octlevel.max().item()-cfg.model.outside_level
    grid_voxel_coord = ((voxel_model.vox_center- voxel_model.vox_size * 0.5)-(voxel_model.scene_center-voxel_model.inside_extent*0.5))/voxel_model.inside_extent*(2**max_voxel_level) #복셀의 좌표
    grid_voxel_size = (voxel_model.vox_size / voxel_model.inside_extent) * (2**max_voxel_level) #복셀의 크기
    print(f"grid_voxel_coord = {grid_voxel_coord}") #복셀의 좌표
    print(f"grid_voxel_size = {grid_voxel_size}") #복셀의 크기
    '''
    rand_idx = torch.randperm(len(voxel_model.grid_keys), device='cuda')[:10]
    for i in range(10):
        key = voxel_model.grid_keys[rand_idx[i]]
        grid_res = 2**max_voxel_level
        x= key %grid_res
        y= (key // grid_res) % grid_res
        z= key // (grid_res * grid_res)
        print(f"grid_keys[{rand_idx[i]}] = {key} => ({x}, {y}, {z})") #grid_keys의 좌표 출력
        voxel_coord = grid_voxel_coord[voxel_model.grid2voxel[rand_idx[i]]] #grid_voxel_coord에서 grid2voxel로 복셀 좌표를 얻음
        print(f"grid_voxel_coord[{rand_idx[i]}] = {voxel_coord}") #복셀 좌표 출력
        voxel_size = grid_voxel_size[voxel_model.grid2voxel[rand_idx[i]]] #grid_voxel_size에서 grid2voxel로 복셀 크기를 얻음
        print(f"grid_voxel_size[{rand_idx[i]}] = {voxel_size}") #복
    '''
    for iteration in iter_rng:

        # Start processing time tracking of this iteration
        iter_start.record()

        # Increase the degree of SH by one up to a maximum degree
        if iteration % 1000 == 0:
            voxel_model.sh_degree_add1()

        # Recompute sh from cameras
        if iteration in cfg.procedure.reset_sh_ckpt:
            print("Reset sh0 from cameras.")
            print("Reset shs to zero.")
            voxel_model.reset_sh_from_cameras(tr_cams)
            torch.cuda.empty_cache()

        # Use default super-sampling option
        if iteration > 1000:
            if cfg.regularizer.ss_aug_max > 1:
                tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
            elif 'ss' in tr_render_opt:
                tr_render_opt.pop('ss')  # Use default ss

        need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
        need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
        need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss
        tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth
        #blending weight의 분포를 더 집중되게 유도
        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist
        #blending weight가 ray상에서 depth순서대로 증가하도록 유도
        if iteration >= cfg.regularizer.ascending_from and cfg.regularizer.lambda_ascending:
            tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending

        # Update auto exposure
        if cfg.auto_exposure.enable and iteration in cfg.procedure.auto_exposure_upd_ckpt:
            for cam in tr_cams:
                with torch.no_grad():
                    ref = voxel_model.render(cam, ss=1.0)['color']
                cam.auto_exposure_update(ref, cam.image.cuda())

        # Pick a Camera
        cam = tr_cams[tr_cam_indices[iteration-1]]

        # Get gt image
        gt_image = cam.image.cuda()
        if cfg.regularizer.lambda_R_concen > 0:
            tr_render_opt['gt_color'] = gt_image

        # Render 결과를 얻는 부분분
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        render_image = render_pkg['color'] #rendered image

        # Loss
        mse = loss_utils.l2_loss(render_image, gt_image)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        loss = cfg.regularizer.lambda_photo * photo_loss #1. mse loss

        if need_sparse_depth: #sparse depth loss 추가 실험용
            loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)

        if cfg.regularizer.lambda_mask: #추가 실험용
            gt_T = 1 - cam.mask.cuda()
            loss += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)

        if need_depthanythingv2: #추가 실험용
            loss += cfg.regularizer.lambda_depthanythingv2 * depthanythingv2_loss(cam, render_pkg, iteration)

        if need_mast3r_metric_depth: #추가 실험용
            loss += cfg.regularizer.lambda_mast3r_metric_depth * mast3r_metric_depth_loss(cam, render_pkg, iteration)

        if cfg.regularizer.lambda_ssim: #2. SSIM loss
            loss += cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
        if cfg.regularizer.lambda_T_concen: #3. concentration Transmittance loss(final이 0or1)
            loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg[f'raw_T'])
        if cfg.regularizer.lambda_T_inside: #to encourage T to be inside the bounding box
            loss += cfg.regularizer.lambda_T_inside * render_pkg[f'raw_T'].square().mean()
        if need_nd_loss: #mesh 용도 normal dmean loss(인접 픽셀 뎁스차이로 구한 depth vs render_pkg['normal']->ray 내에서 weight rendering)
            loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
        if need_nmed_loss: #mesh 용도 normal dmed loss(median depth vs render_pkg['normal']->ray 내에서 weight rendering)
            loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)
        # lambda_R_concen loss는 render_opt에 넣어줘야함 (픽셀컬러랑 ray 복셀 컬러 차이 l2 loss)
        # lambda_dist loss는 render_opt에 넣어줘야함 (ray에서 얼마나 density가 모여있는지)
        # lambda_ascending loss는 render_opt에 넣어줘야함
        # lambda tv loss는 voxel_model.optimizer.step() 이후에 적용해야함 (복셀간의 smoothness)
        # Backward to get gradient of current iteration
        voxel_model.optimizer.zero_grad(set_to_none=True)

        





        loss.backward()

        # Grid-level regularization
        grid_reg_interval = iteration >= cfg.regularizer.tv_from and iteration <= cfg.regularizer.tv_until
        if cfg.regularizer.lambda_tv_density and grid_reg_interval:
            lambda_tv_mult = cfg.regularizer.tv_decay_mult ** (iteration // cfg.regularizer.tv_decay_every)
            svraster_cuda.grid_loss_bw.total_variation(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                weight=cfg.regularizer.lambda_tv_density * lambda_tv_mult,
                vox_size_inv=voxel_model.vox_size_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.tv_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)

        voxel_gradient_interval = iteration >= cfg.regularizer.vg_from and iteration <= cfg.regularizer.vg_until
        if cfg.regularizer.lambda_vg_density and voxel_gradient_interval:
            lambda_vg_mult = cfg.regularizer.vg_decay_mult ** (iteration // cfg.regularizer.vg_decay_every)
            svraster_cuda.grid_loss_bw.voxel_gradient(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                weight=cfg.regularizer.lambda_vg_density * lambda_vg_mult,
                vox_size_inv=voxel_model.vox_size_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.vg_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)
        
        grid_eikonal_interval = iteration >= cfg.regularizer.ge_from and iteration <= cfg.regularizer.ge_until
        if cfg.regularizer.lambda_ge_density and grid_eikonal_interval:
            lambda_ge_mult = cfg.regularizer.ge_decay_mult ** (iteration // cfg.regularizer.ge_decay_every)
            svraster_cuda.grid_loss_bw.grid_eikonal(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                grid_voxel_coord=grid_voxel_coord,
                grid_voxel_size=grid_voxel_size.view(-1),
                grid_res= 2**max_voxel_level,
                grid_mask=voxel_model.grid_mask,
                grid_keys= voxel_model.grid_keys,
                grid2voxel=voxel_model.grid2voxel,
                weight=cfg.regularizer.lambda_ge_density * lambda_ge_mult,
                vox_size_inv=vox_size_min_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.ge_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)
        
        
        # Optimizer step
        voxel_model.optimizer.step()  # SVOptimizer
        
        # Learning rate warmup scheduler step
        if iteration <= cfg.optimizer.n_warmup: #learning rate warmup
            rate = iteration / cfg.optimizer.n_warmup
            for param_group in voxel_model.optimizer.param_groups:
                param_group["lr"] = rate * param_group["base_lr"]

        if iteration in cfg.optimizer.lr_decay_ckpt: #learning rate decay
            for param_group in voxel_model.optimizer.param_groups:
                ori_lr = param_group["lr"]
                param_group["lr"] *= cfg.optimizer.lr_decay_mult
                print(f'LR decay of {param_group["name"]}: {ori_lr} => {param_group["lr"]}')
            cfg.regularizer.lambda_vg_density *= cfg.optimizer.lr_decay_mult
            cfg.regularizer.lambda_tv_density *= cfg.optimizer.lr_decay_mult
            cfg.regularizer.lambda_ge_density *= cfg.optimizer.lr_decay_mult
        '''
        if( cfg.model.density_mode == 'sdf' and iteration == 10000):
            cfg.regularizer.lambda_vg_density /= 10.0
            cfg.regularizer.lambda_tv_density /= 10.0
        '''
        ######################################################
        # Gradient statistic should happen before adaptive op
        ######################################################

        need_stat = (
            iteration >= 500 and \
            iteration <= cfg.procedure.subdivide_until)
        if need_stat:
            voxel_model.subdiv_meta += voxel_model._subdiv_p.grad 
            # 각 voxel마다 gradient 크기를 저장 -> subdivide할 때 사용

        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        need_pruning = ( #prune_every = 1000
            iteration % cfg.procedure.prune_every == 0 and \
            iteration >= cfg.procedure.prune_from and \
            iteration <= cfg.procedure.prune_until)
        need_subdividing = ( #subdivide_every = 1000
            iteration % cfg.procedure.subdivide_every == 0 and \
            iteration >= cfg.procedure.subdivide_from and \
            iteration <= cfg.procedure.subdivide_until and \
            voxel_model.num_voxels < cfg.procedure.subdivide_max_num)

        # Do nothing in last 500 iteration
        need_pruning &= (iteration <= cfg.procedure.n_iter-500)
        need_subdividing &= (iteration <= cfg.procedure.n_iter-500)

        if need_pruning or need_subdividing:
            stat_pkg = voxel_model.compute_training_stat(camera_lst=tr_cams)
            torch.cuda.empty_cache()
        # max_w = stat_pkg['max_w'] # 각 voxel마다 max weight를 저장 (T_i*a_i)
        # min_samp_interval = stat_pkg['min_samp_interval'] # 각 voxel마다 min sampling interval을 저장 (T_i*a_i)
        # view_cnt = stat_pkg['view_cnt'] # 각 voxel마다 view count를 저장
        if need_pruning:
            ori_n = voxel_model.num_voxels

            # Compute pruning threshold
            prune_all_iter = max(1, cfg.procedure.prune_until - cfg.procedure.prune_every)
            prune_now_iter = max(0, iteration - cfg.procedure.prune_every)
            prune_iter_rate = max(0, min(1, prune_now_iter / prune_all_iter))
            thres_inc = max(0, cfg.procedure.prune_thres_final - cfg.procedure.prune_thres_init)
            prune_thres = cfg.procedure.prune_thres_init + thres_inc * prune_iter_rate

            # Prune voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)
            if cfg.model.density_mode == 'sdf' and iteration >= 1000:
                sdf_vals = voxel_model._geo_grid_pts[voxel_model.vox_key]  # [N, 8, 1]
                signs = (sdf_vals > 0).float()
                has_surface = (signs.min(dim=1).values != signs.max(dim=1).values).view(-1)

                min_abs_sdf = sdf_vals.abs().min(dim=1).values.view(-1)  # 각 복셀의 가장 가까운 꼭짓점 SDF 절댓값
                global_vox_size_min = voxel_model.vox_size.min().item()
                sdf_thresh = 3.0 * global_vox_size_min
                # sdf_thresh 감소: 3000 → 15000에서 0.5 → 0.2
                '''
                start_iter = 3000
                end_iter = 15000
                start_thresh = 0.5
                end_thresh = 0.2
                progress = min(1.0, max(0.0, (iteration - start_iter) / (end_iter - start_iter)))
                sdf_thresh = start_thresh + (end_thresh - start_thresh) * progress
                '''
                prune_mask = (~has_surface) & (min_abs_sdf > sdf_thresh) 
            voxel_model.pruning(prune_mask)

            # Prune statistic (for the following subdivision)
            kept_idx = (~prune_mask).argwhere().squeeze(1)
            for k, v in stat_pkg.items():
                stat_pkg[k] = v[kept_idx] #v는 원래 [N, 1]짜리였던 통계 tensor, 필터로 pruning

            print(f"voxel_model._log_s = {voxel_model._log_s}")

            new_n = voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')
            torch.cuda.empty_cache()

        if need_subdividing:
            # Exclude some voxels
            size_thres = stat_pkg['min_samp_interval'] * cfg.procedure.subdivide_samp_thres
            large_enough_mask = (voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest_mask = voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS 
            if cfg.model.density_mode == 'sdf':
                non_finest_mask = voxel_model.octlevel.squeeze(1) < (svraster_cuda.meta.MAX_NUM_LEVELS-2)
            valid_mask = large_enough_mask & non_finest_mask

            # Get some statistic for subdivision priority
            priority = voxel_model.subdiv_meta.squeeze(1) * valid_mask

            # Compute priority rank (larger value has higher priority)
            rank = torch.zeros_like(priority)
            rank[priority.argsort()] = torch.arange(len(priority), dtype=torch.float32, device="cuda")

            # Determine the number of voxels to subdivided
            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = rank.quantile(1 - subdivide_prop)
            
            # Compute subdivision mask
            
            subdivide_mask = (rank > thres) & valid_mask

            if cfg.model.density_mode == 'sdf': #수정해라
                with torch.no_grad():
                    sdf_vals = voxel_model._geo_grid_pts[voxel_model.vox_key]  # [N, 8, 1]
                    signs = (sdf_vals > 0).float()
                    has_surface = (signs.min(dim=1).values != signs.max(dim=1).values).view(-1)

                cur_level = voxel_model.octlevel.squeeze(1)
                max_level = 9 +cfg.model.outside_level # 9 + 1 = 10
                under_level = cur_level < max_level
                valid_mask = has_surface & under_level

                if iteration < 15000:
                    topk = 50000

                    valid_indices = torch.where(valid_mask)[0]
                    
                    valid_priorities = priority[valid_mask]

                    k = min(topk, valid_priorities.numel())
                    topk_rel_idx = torch.topk(valid_priorities, k).indices
                    topk_abs_idx = valid_indices[topk_rel_idx]

                    subdivide_mask = torch.zeros_like(priority, dtype=torch.bool)
                    subdivide_mask[topk_abs_idx] = True
                else:
                    # 15k 이후는 조건에 맞는 모든 voxel subdivision
                    subdivide_mask = valid_mask
            print(f"voxel_model._log_s = {voxel_model._log_s}")
            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (rank > rank[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            ori_n = voxel_model.num_voxels
            if subdivide_mask.sum() > 0:
                voxel_model.subdividing(subdivide_mask, cfg.procedure.subdivide_save_gpu)
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            voxel_model.subdiv_meta.zero_()  # reset subdiv meta
            remain_subdiv_times -= 1
            torch.cuda.empty_cache()
        if need_pruning or need_subdividing:
            vox_size_min_inv = 1.0 / voxel_model.vox_size.min().item() #복셀의 최소 크기
            #print(f"voxel_model.vox_size_min_inv = {vox_size_min_inv:.4f}") #복셀의 최소 크기
            max_voxel_level = voxel_model.octlevel.max().item()-cfg.model.outside_level
            grid_voxel_coord = ((voxel_model.vox_center- voxel_model.vox_size * 0.5)-(voxel_model.scene_center-voxel_model.inside_extent*0.5))/voxel_model.inside_extent*(2**max_voxel_level) #복셀의 좌표
            grid_voxel_size = (voxel_model.vox_size / voxel_model.inside_extent) * (2**max_voxel_level) #복셀의 크기
            #print(f"grid_voxel_coord = {grid_voxel_coord}") #복셀의 좌표
            #print(f"grid_voxel_size = {grid_voxel_size}") #복셀의 크기
            print(f'level max : {max_voxel_level}') #복셀의 최대 레벨
            #print(f"voxel_model.vox_size_min_inv = {vox_size_min_inv:.4f}") #복셀의 최소 크기

            voxel_model.grid_mask, voxel_model.grid_keys, voxel_model.grid2voxel = octree_utils.update_valid_gradient_table(cfg.model.density_mode, voxel_model.vox_center, voxel_model.vox_size, voxel_model.scene_center, voxel_model.inside_extent, max_voxel_level)
            torch.cuda.synchronize()
            '''
            print(f"voxel_model.grid_mask = {voxel_model.grid_mask}") #grid_mask 출력
            rand_idx = torch.randperm(len(voxel_model.grid_keys), device='cuda')[:10]
            grid_res = 2 ** max_voxel_level
            grid_mask = voxel_model.grid_mask
            grid_keys = voxel_model.grid_keys
            grid2voxel = voxel_model.grid2voxel

            for i in range(10):
                key = grid_keys[rand_idx[i]].item()
                x = key % grid_res
                y = (key // grid_res) % grid_res
                z = key // (grid_res * grid_res)
                print(f"\n[{i}] key = {key} => ({x}, {y}, {z})")

                voxel_id = grid2voxel[rand_idx[i]].item()
                voxel_coord = grid_voxel_coord[voxel_id]
                voxel_size = grid_voxel_size[voxel_id]
                print(f"Voxel ID: {voxel_id}, Coord: {voxel_coord}, Size: {voxel_size}")

                def flatten(x, y, z):
                    return x + y * grid_res + z * grid_res * grid_res

                directions = [("x+1", 1, 0, 0), ("x-1", -1, 0, 0),
                            ("y+1", 0, 1, 0), ("y-1", 0, -1, 0),
                            ("z+1", 0, 0, 1), ("z-1", 0, 0, -1)]

                for label, dx, dy, dz in directions:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < grid_res and 0 <= ny < grid_res and 0 <= nz < grid_res:
                        nkey = flatten(nx, ny, nz)
                        in_mask = grid_mask[nkey].item()
                        in_keys = nkey in grid_keys
                        print(f"  Neighbor {label} ({nx}, {ny}, {nz}) key={nkey} -> "
                            f"mask={in_mask}, binary_search_hit={in_keys}")
                    else:
                        print(f"  Neighbor {label} out-of-bounds")'''
        ######################################################
        # End of adaptive voxels procedure
        ######################################################

        # End processing time tracking of this iteration
        iter_end.record()
        torch.cuda.synchronize()
        elapsed += iter_start.elapsed_time(iter_end)

        # Logging
        with torch.no_grad():
            # Metric
            loss = loss.item()
            psnr = -10 * np.log10(mse.item())

            # Progress bar
            ema_p = max(0.01, 1 / (iteration - first_iter + 1))
            ema_loss_for_log += ema_p * (loss - ema_loss_for_log)
            ema_psnr_for_log += ema_p * (psnr - ema_psnr_for_log)
            if iteration % 10 == 0:
                pb_text = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                }
                progress_bar.set_postfix(pb_text)
                progress_bar.update(10)
            if iteration == cfg.procedure.n_iter:
                progress_bar.close()

            # Log and save `
            
            training_report(
                data_pack=data_pack,
                voxel_model=voxel_model,
                iteration=iteration,
                loss=loss,
                psnr=psnr,
                elapsed=elapsed,
                ema_psnr=ema_psnr_for_log,
                pg_view_every=args.pg_view_every,
                test_iterations=args.test_iterations) 

            if iteration in args.checkpoint_iterations or iteration == cfg.procedure.n_iter:
                voxel_model.save_iteration(iteration, quantize=args.save_quantized)
                if args.save_optimizer:
                    voxel_model.optimizer_save_iteration(iteration)
                print(f"[SAVE] path={voxel_model.latest_save_path}")


def training_report(data_pack, voxel_model, iteration, loss, psnr, elapsed, ema_psnr, pg_view_every, test_iterations):

    voxel_model.freeze_vox_geo()
    if pg_view_every > 0 and (iteration % pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = 0
        view = test_cameras[pg_idx]

        # render: only color (disable depth, normal, T)
        render_pkg = voxel_model.render(view)

        render_image = render_pkg['color']
        im = im_tensor2np(render_image)  # Just RGB image

        torch.cuda.empty_cache()

        outdir = os.path.join(voxel_model.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)
        imageio.imwrite(outpath, im)

        eps_file = os.path.join(voxel_model.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n") 
    voxel_model.unfreeze_vox_geo()
    
    # Progress view
    if pg_view_every > 0 and (iteration % pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = 0
        view = test_cameras[pg_idx]
        render_pkg = voxel_model.render(view, output_depth=True, output_normal=True, output_T=True)
        render_image = render_pkg['color']
        render_depth = render_pkg['depth'][0]
        render_depth_med = render_pkg['depth'][2]
        render_normal = render_pkg['normal']
        render_alpha = 1 - render_pkg['T'][0]

        im = np.concatenate([
            np.concatenate([
                im_tensor2np(render_image),
                im_tensor2np(render_alpha)[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                viz_tensordepth(render_depth, render_alpha),
                im_tensor2np(render_normal * 0.5 + 0.5),
            ], axis=1),
            np.concatenate([
                im_tensor2np(view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(view.depth2normal(render_depth_med) * 0.5 + 0.5),
            ], axis=1),
        ], axis=0)
        torch.cuda.empty_cache()

        outdir = os.path.join(voxel_model.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)

        imageio.imwrite(outpath, im)

        eps_file = os.path.join(voxel_model.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n")
   
    # Report test and samples of training set
    if iteration in test_iterations:
        print(f"[EVAL] running...")
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        save_every = max(1, len(test_cameras) // 8)
        outdir = os.path.join(voxel_model.model_path, "test_view")
        os.makedirs(outdir, exist_ok=True)
        psnr_lst = []
        video = []
        max_w = torch.zeros([voxel_model.num_voxels, 1], dtype=torch.float32, device="cuda")
        for idx, camera in enumerate(test_cameras):
            render_pkg = voxel_model.render(camera, output_normal=True, track_max_w=True)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            video.append(im)
            if idx % save_every == 0:
                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}.jpg")
                cat = np.concatenate([gt, im], axis=1)
                imageio.imwrite(outpath, cat)

                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}_normal.jpg")
                render_normal = render_pkg['normal']
                render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
                imageio.imwrite(outpath, render_normal)
            mse = np.square(im/255 - gt/255).mean()
            psnr_lst.append(-10 * np.log10(mse))
            max_w = torch.maximum(max_w, render_pkg['max_w'])
        avg_psnr = np.mean(psnr_lst)
        imageio.mimwrite(
            os.path.join(outdir, f"video_iter{iteration:06d}.mp4"),
            video, fps=30)
        torch.cuda.empty_cache()

        fps = time.time()
        for idx, camera in enumerate(test_cameras):
            voxel_model.render(camera, track_max_w=False)
        torch.cuda.synchronize()
        fps = len(test_cameras) / (time.time() - fps)
        torch.cuda.empty_cache()

        # Sample training views to render
        train_cameras = data_pack.get_train_cameras()
        for idx in range(0, len(train_cameras), max(1, len(train_cameras)//8)):
            camera = train_cameras[idx]
            render_pkg = voxel_model.render(
                camera, output_normal=True, track_max_w=True,
                use_auto_exposure=cfg.auto_exposure.enable)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}.jpg")
            cat = np.concatenate([gt, im], axis=1)
            imageio.imwrite(outpath, cat)

            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}_normal.jpg")
            render_normal = render_pkg['normal']
            render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
            imageio.imwrite(outpath, render_normal)

        print(f"[EVAL] iter={iteration:6d}  psnr={avg_psnr:.2f}  fps={fps:.0f}")

        outdir = os.path.join(voxel_model.model_path, "test_stat")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.json")
        os.makedirs(outdir, exist_ok=True)
        with open(outpath, 'w') as f:
            q = torch.linspace(0,1,5, device="cuda")
            max_w_q = max_w.quantile(q).tolist()
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            stat = {
                'psnr': avg_psnr,
                'ema_psnr': ema_psnr,
                'elapsed': elapsed,
                'fps': fps,
                'n_voxels': voxel_model.num_voxels,
                'max_w_q': max_w_q,
                'peak_mem': peak_mem,
            }
            json.dump(stat, f, indent=4)
    



if __name__ == "__main__":

    # Parse arguments (command line)
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster optimization."
        "You can specify a list of config files to overwrite the default setups."
        "All config fields can also be overwritten by command line.")
    parser.add_argument('--cfg_files', default=[], nargs='*')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[-1])
    parser.add_argument("--pg_view_every", type=int, default=200)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--load_optimizer", action='store_true')
    parser.add_argument("--save_optimizer", action='store_true')
    parser.add_argument("--save_quantized", action='store_true')
    parser.add_argument("--seunghun", action='store_true', help="Enable Seunghun's custom settings.") #seunghun

    args, cmd_lst = parser.parse_known_args()

    # Update config from files and command line
    update_config(args.cfg_files, cmd_lst)
    
    if args.seunghun:
        cfg.model.density_mode = "sdf"
        cfg.model.vox_geo_mode = "triinterp3"
        cfg.optimizer.geo_lr = 0.0025
        cfg.optimizer.lr_decay_ckpt = [7000,14000]
        cfg.optimizer.lr_decay_mult = 0.3
        cfg.init.geo_init = 0.0
        cfg.regularizer.dist_from = 10000
        cfg.regularizer.lambda_dist = 0.1
        cfg.regularizer.lambda_tv_density = 0.0 #1e-6
        cfg.regularizer.tv_from = 0000
        cfg.regularizer.tv_until = 20000
        cfg.regularizer.lambda_vg_density = 0.0 #1e-8
        cfg.regularizer.vg_until = 20000
        cfg.regularizer.lambda_ascending = 0.0
        cfg.regularizer.ascending_from = 0
        cfg.regularizer.lambda_normal_dmean = 0.0
        cfg.regularizer.n_dmean_from = 20000  # 이거 넣으면 터짐 왜지?
        cfg.regularizer.lambda_normal_dmed = 0.0
        cfg.regularizer.n_dmed_from = 10000
        cfg.procedure.prune_from = 3000
        cfg.procedure.prune_every = 3000
        cfg.procedure.prune_until = 15000
        cfg.procedure.subdivide_from = 3000
        cfg.procedure.subdivide_every = 3000

    # Global init
    seed_everything(cfg.procedure.seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Setup output folder and dump config
    if not cfg.model.model_path:
        datetime_str = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
        unique_str = str(uuid.uuid4())[:6]
        folder_name = f"{datetime_str}-{unique_str}"
        cfg.model.model_path = os.path.join(f"./output", folder_name)

    os.makedirs(cfg.model.model_path, exist_ok=True)
    with open(os.path.join(cfg.model.model_path, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    print(f"Output folder: {cfg.model.model_path}")

    # Apply scheduler scaling
    if cfg.procedure.sche_mult != 1:
        sche_mult = cfg.procedure.sche_mult

        for key in ['geo_lr', 'sh0_lr', 'shs_lr']:
            cfg.optimizer[key] /= sche_mult
        cfg.optimizer.n_warmup = round(cfg.optimizer.n_warmup * sche_mult)
        cfg.optimizer.lr_decay_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.optimizer.lr_decay_ckpt]

        for key in [
                'dist_from', 'tv_from', 'tv_until',
                'n_dmean_from', 'n_dmean_end',
                'n_dmed_from', 'n_dmed_end',
                'depthanythingv2_from', 'depthanythingv2_end',
                'mast3r_metric_depth_from', 'mast3r_metric_depth_end']:
            cfg.regularizer[key] = round(cfg.regularizer[key] * sche_mult)

        for key in [
                'n_iter',
                'prune_from', 'prune_every', 'prune_until',
                'subdivide_from', 'subdivide_every', 'subdivide_until']:
            cfg.procedure[key] = round(cfg.procedure[key] * sche_mult)
        cfg.procedure.reset_sh_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.procedure.reset_sh_ckpt] #scale 조정

    # Update negative iterations
    for i in range(len(args.test_iterations)):
        if args.test_iterations[i] < 0:
            args.test_iterations[i] += cfg.procedure.n_iter + 1
    for i in range(len(args.checkpoint_iterations)):
        if args.checkpoint_iterations[i] < 0:
            args.checkpoint_iterations[i] += cfg.procedure.n_iter + 1

    # Launch training loop
    training(args)
    print("Everything done.")
