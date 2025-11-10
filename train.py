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
from src.dataloader.data_pack import DataPack, compute_iter_idx_sparse
from src.sparse_voxel_model import SparseVoxelModel

import svraster_cuda


def training(args): 
    # Init and load data pack
    data_pack = DataPack(cfg.data, cfg.model.white_background) #data_pack.py -> DataPack

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    print(f"Training view num = {len(tr_cams)}")
    tr_cam_indices = compute_iter_idx_sparse(len(tr_cams), cfg.procedure.n_iter, 1) #data_pack.py -> compute_iter_idx

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init() 
    

    # Prepare monocular depth priors if instructed
    if cfg.regularizer.lambda_depthanythingv2:
        mono_utils.prepare_depthanythingv2(
            cameras=tr_cams,
            source_path=os.path.dirname(cfg.data.source_path), # modified
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
    ) #  inner bounding box
    

    # Init voxel model
    voxel_model = SparseVoxelModel(cfg.model)
    
    if args.load_iteration: 
        loaded_iter = voxel_model.load_iteration(args.load_iteration)
    else:
        loaded_iter = None
        voxel_model.model_init( #from constructor.py 
            bounding=bounding,
            cfg_init=cfg.init,
            cfg_mode= cfg.model.density_mode,
            cameras=tr_cams,
            sfm_init=data_pack.sfm_init_data
        )
    

    initial_points = torch.from_numpy(data_pack.sfm_init_data.points_xyz).float().to("cuda")
    print(f"point num = {initial_points.shape[0]}")
    print(initial_points)


    first_iter = loaded_iter if loaded_iter else 1
    print(f"Start optmization from iters={first_iter}.") 

    # Init optimizer
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
    #Subdiv: times=15 scale-each-time=135.0% prop=5.0% 占쎌녃域뱄퐦�삕�뜝�룞�삕�뜮占�
    # 15?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� subdivide, 癲ル슢�뵞占쎄콪占쎈뼀�뜝占� 135% scale 癲ル슣鍮섌뜮占썲뜝�럩援�?? #5/10
    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True) #execute time 癲ル쉵���猷긷뜝�럩�젳 in GPU
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0 #elapsed time

    tr_render_opt = {
        'track_max_w': False, # track max_w for subdivision
        'lambda_R_concen': cfg.regularizer.lambda_R_concen, # lambda_R_concen loss
        'output_T': False, # ray marching 濚욌꼬�궡�꺁占쏙옙�슪�삕 占쎈눇�뙼�맪占쎌���삕亦낉옙 �뜝�럥�돯�뜝�럥痢듿뜝�럩議� T
        'output_depth': False, #true ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� depth �뜝�럥�돯�뜝�럥痢듿뜝�럩議�
        'ss': 1.0,  # disable supersampling at first super-sampling ratio
        'rand_bg': cfg.regularizer.rand_bg, # random background color
        'use_auto_exposure': cfg.auto_exposure.enable, # use auto exposure
    }
    #占쎈뙑��⑤슦占쎌��琉껈겫猷몄맶占쎈쐻�뜝占�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
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
    pi3_normal_loss = loss_utils.Pi3NormalLoss(
        iter_from=cfg.regularizer.pi3_normal_from,
        iter_end=cfg.regularizer.pi3_normal_end
    )

    ema_loss_for_log = 0.0 #use for logging
    ema_psnr_for_log = 0.0 #use for logging
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training")   #10 iterations癲ル슢�뵯占쎌맄�뜝�럥�렡 processbar update
    #parameters for eikonal loss
    vox_size_min_inv = 1.0 / voxel_model.vox_size.min().item() #占쎌녃域뱄퐢荑덂뜝�럩援�?? 癲ル슔�걠獒뺣돍�삕�댆占� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
    #print(f"voxel_model.vox_size_min_inv = {vox_size_min_inv:.4f}") #占쎌녃域뱄퐢荑덂뜝�럩援�?? 癲ル슔�걠獒뺣돍�삕�댆占� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
    max_voxel_level = voxel_model.octlevel.max().item()-cfg.model.outside_level
    grid_voxel_coord = ((voxel_model.vox_center- voxel_model.vox_size * 0.5)-(voxel_model.scene_center-voxel_model.inside_extent*0.5))/voxel_model.inside_extent*(2**max_voxel_level) #占쎌녃域뱄퐢荑덂뜝�럩援�?? �뜝�럥苑욃뜝�럩�뮡嶺뚮쪋�삕
    grid_voxel_size = (voxel_model.vox_size / voxel_model.inside_extent) * (2**max_voxel_level) #占쎌녃域뱄퐢荑덂뜝�럩援�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�??
    #print(f"grid_voxel_coord = {grid_voxel_coord}") #占쎌녃域뱄퐢荑덂뜝�럩援�?? �뜝�럥苑욃뜝�럩�뮡嶺뚮쪋�삕
    #print(f"grid_voxel_size = {grid_voxel_size}") #占쎌녃域뱄퐢荑덂뜝�럩援�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�??

    #initial the logs value, criteria=99% 
    device = voxel_model._log_s.device
    dtype  = voxel_model._log_s.dtype
    learning_thickness = 2.0
    vsmi = torch.as_tensor(vox_size_min_inv, device=device, dtype=dtype)
    init = 0.1 * torch.log(
        torch.log(torch.tensor(99.0, device=device, dtype=dtype)) * vsmi / learning_thickness/2
    )

    with torch.no_grad():
        voxel_model._log_s.copy_(init)   # <<<< 占쎈쐻占쎈윪占쎄땍占쎈쎗占쎈즴占쎈씔�뜝�럥�뜲占쎈쐻占쎈윥占쎈윝 占쎈섀占쏙옙占쏙옙�맆占쎈쐻占쎈짗占쎌굲! 占쎈쨬占쎈즴�뜝�뜦維쀧빊占� 占쎌녃域뱄퐢苡답벧�굢�삕

    print(f"log_s init = {voxel_model._log_s.item():.9f}")
    '''
    rand_idx = torch.randperm(len(voxel_model.grid_keys), device='cuda')[:10]
    for i in range(10):
        key = voxel_model.grid_keys[rand_idx[i]]
        grid_res = 2**max_voxel_level
        x= key %grid_res
        y= (key // grid_res) % grid_res
        z= key // (grid_res * grid_res)
        print(f"grid_keys[{rand_idx[i]}] = {key} => ({x}, {y}, {z})") #grid_keys?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� �뜝�럥苑욃뜝�럩�뮡嶺뚮쪋�삕 �뜝�럥�돯�뜝�럥痢듿뜝�럩議�
        voxel_coord = grid_voxel_coord[voxel_model.grid2voxel[rand_idx[i]]] #grid_voxel_coord?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� grid2voxel�뜝�럥�맶占쎈쐻�뜝占�??? 占쎌녃域뱄퐢荑덂뜝�럩援�?? �뜝�럥苑욃뜝�럩�뮡嶺뚮ㅏ�뼠占쎌맶占쎈쐻�뜝占�??? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
        print(f"grid_voxel_coord[{rand_idx[i]}] = {voxel_coord}") #占쎌녃域뱄퐢荑덂뜝�럩援�?? �뜝�럥苑욃뜝�럩�뮡嶺뚮쪋�삕 �뜝�럥�돯�뜝�럥痢듿뜝�럩議�
        voxel_size = grid_voxel_size[voxel_model.grid2voxel[rand_idx[i]]] #grid_voxel_size?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� grid2voxel�뜝�럥�맶占쎈쐻�뜝占�??? 占쎌녃域뱄퐢荑덂뜝�럩援�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援뀐옙堉②퐛紐껓옙占썲뜝�럩援�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
        print(f"grid_voxel_size[{rand_idx[i]}] = {voxel_size}") #�뜝�럥�맶占쎈쐻�뜝占�???
    '''
    #initial prune
    

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

        #increase the std value
        '''
        first_prune = cfg.procedure.prune_from
        prune_every = cfg.procedure.prune_every
        std_increase_rate = 0.07/(prune_every*2)
        if first_prune <= iteration <= (first_prune + prune_every*6):
            with torch.no_grad():
                voxel_model._log_s.add_(std_increase_rate)
        '''
        
        if not hasattr(cfg.regularizer, "_ge_final"):
            cfg.regularizer._ge_final = float(cfg.regularizer.lambda_ge_density)
        if not hasattr(cfg.regularizer, "_ls_final"):
            cfg.regularizer._ls_final = float(cfg.regularizer.lambda_ls_density)
        '''
        # 占쎈뮞��놂옙餓ο옙 占쎌읅占쎌뒠
        if iteration <1000:
            cfg.regularizer.lambda_ge_density = 0.0
            cfg.regularizer.lambda_ls_density = 0.0
        elif iteration == 1000:
            cfg.regularizer.lambda_ge_density = cfg.regularizer._ge_final
            cfg.regularizer.lambda_ls_density = cfg.regularizer._ls_final
        
        
        if 2000<=iteration <= 8000:
            s = 0.25+ 0.75 * ((iteration%2000) / 2000.0)  # 0.1占쎈꼥1.0
            if iteration <= 2000:
                cfg.regularizer.lambda_ge_density = s * cfg.regularizer._ge_final
            cfg.regularizer.lambda_ls_density = s * cfg.regularizer._ls_final
        '''
        if iteration == 10000:
        # 占쎌긿占쎈뼒筌띾뜆�뵠占쏙옙占쏙옙�벥 筌뤴뫀諭� 占쎈솁占쎌뵬沃섎챸苑� 域밸챶竊숋옙�뱽 占쎈떄占쎌돳
        # self.optimizer -> voxel_model.optimizer 嚥∽옙 占쎈땾占쎌젟
            for param_group in voxel_model.optimizer.param_groups:
                # 占쎌뵠�뵳袁⑹뵠 'log_s'占쎌뵥 域밸챶竊숋옙�뱽 筌≪뼚釉섓옙苑� lr占쎌뱽 癰귨옙野껓옙
                if param_group['name'] == 'log_s':
                    # self.cfg_optimizer -> cfg.optimizer 嚥∽옙 占쎈땾占쎌젟
                    target_lr = cfg.optimizer.log_s_lr # 占쎄퐬占쎌젟 占쎈솁占쎌뵬占쎈퓠 占쎌젟占쎌벥占쎈쭆 揶쏉옙
                    param_group['lr'] = target_lr
                    print(f"\n[INFO] Iteration {iteration}: `log_s` learning rate changed to {target_lr}\n")
                    break # 筌≪뼚釉�占쎌몵占쎈빍 �뙴�뫂遊� �넫�굝利�

        first_prune = cfg.procedure.prune_from
        prune_every = cfg.procedure.prune_every
        std_increase_rate = 0.07 / (prune_every * 2)

        # 鈺곌퀗援뷂옙�뱽 first_prune�겫占쏙옙苑� 8000 沃섎챶彛붹틦�슣占쏙옙嚥∽옙 癰귨옙野껓옙
        if 1 <= iteration < 10000: # 疫꿸퀣���: <= (first_prune + prune_every*6)
            with torch.no_grad():
                voxel_model._log_s.add_(std_increase_rate)


        if(iteration %100 ==0):
            print(f"iteration {iteration} log_s = {voxel_model._log_s.item():.9f}")
        need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
        need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
        need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        need_pi3_normal_loss = cfg.regularizer.lambda_pi3_normal > 0 and pi3_normal_loss.is_active(iteration)
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss or need_pi3_normal_loss
        tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth
        #blending weight �뜝�럡猿��넭怨κ데塋딆���삕占쎌맶占쎈쐻�뜝占�?? 癲ル슣�돳筌ㅻĿ�뀋�뜝占�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist
        #blending weight ray 占쎈쎗占쎈젻泳��떑�젂占쎄퐩占쎌맶占쎈쐻�뜝占�?? depth 占쎈쎗占쎈젻泳��떑�젂�뜝占�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�?? 癲ル슣鍮섌뜮蹂�占쎈슪�삕?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
        if iteration >= cfg.regularizer.ascending_from and iteration <= cfg.regularizer.ascending_until and cfg.regularizer.lambda_ascending:
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

        # Render 占쎈눇�뙼�맪占쎌���삕亦낉옙
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        render_image = render_pkg['color'] #rendered image

        '''
        # Loss
        mse = loss_utils.l2_loss(render_image, gt_image)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        loss_photo = cfg.regularizer.lambda_photo * photo_loss #1. mse loss
        loss = loss_photo
        loss_dict = {"photo": loss_photo}
        '''
        # Loss
        #gt_mask = cam.mask.cuda() 

        gt_image_modified = gt_image #*gt_mask

        mse = loss_utils.l2_loss(render_image, gt_image_modified)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image_modified)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image_modified, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        
        loss_photo = cfg.regularizer.lambda_photo * photo_loss
        loss = loss_photo
        loss_dict = {"photo": loss_photo}
        # if need_sparse_depth: #sparse depth loss 占쎈툡占쎌뒄占쎈뻻
        #     loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)
        '''
        loss_mask = 0
        if cfg.regularizer.lambda_mask: #筌띾뜆�뮞占쎄쾿 占쎈��占쎈뼄 占쎈툡占쎌뒄占쎈뻻
            gt_T = 1 - cam.mask.cuda()
            loss_mask += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)
            loss+=loss_mask
            loss_dict["mask"] = loss_mask
        '''
        '''
        loss_mask = 0
        if cfg.regularizer.lambda_mask:
            # 1. 筌뤴뫀�쑞占쎌뵠 占쎌굙筌β돧釉� 占쎈떮��⑥눖猷� 筌띾벊�뱽 揶쏉옙占쎌죬占쎌긿占쎈빍占쎈뼄.
            pred_T = render_pkg['T']
            
            # 2. 獄쏄퀗瑗� 筌띾뜆�뮞占쎄쾿�몴占� 占쎄문占쎄쉐占쎈��占쎈빍占쎈뼄. (獄쏄퀗瑗�=1, 揶쏆빘猿�=0)
            # 占쎌뵠 筌띾뜆�뮞占쎄쾿占쎈뮉 '占쎌젟占쎈뼗 T揶쏉옙'占쎌뵠占쎌쁽, '占쎌궎筌△뫀占쏙옙 ��④쑴沅쏉옙釉� 占쎌겫占쎈열'占쎌뱽 筌욑옙占쎌젟占쎈릭占쎈뮉 占쎈열占쎈막占쎌뱽 野껊챸鍮�占쎈빍占쎈뼄.
            background_mask = 1 - cam.mask.cuda()

            # 3. 占쎈동占쏙옙占썼퉪占� 占쎌젫��⑨옙 占쎌궎筌△뫀占쏙옙 ��④쑴沅쏉옙鍮�占쎈빍占쎈뼄. 
            # 獄쏄퀗瑗� 占쎈동占쏙옙占쏙옙�벥 占쎌젟占쎈뼗 T占쎈뮉 1.0占쎌뵠沃섓옙嚥∽옙, (占쎌굙筌β넄而� - 1.0)^2 占쎌굨占쎄묶占쎌벥 占쎌궎筌△몿占쏙옙 ��④쑴沅쏉옙留쀯옙�빍占쎈뼄.
            pixel_error = (pred_T - 1.0) ** 2

            # 4. [占쎈퉹占쎈뼎] 占쎌궎筌∽옙 筌띾벊肉� 獄쏄퀗瑗� 筌띾뜆�뮞占쎄쾿�몴占� ��④퉲鍮�占쎈빍占쎈뼄.
            # 占쎌뵠 占쎈염占쎄텦占쎌뱽 占쎈꽰占쎈퉸 揶쏆빘猿� 占쎌겫占쎈열(background_mask揶쏉옙 0占쎌뵥 �겫占썽겫占�)占쎌벥 占쎌궎筌△뫀�뮉 筌뤴뫀紐� 0占쎌뵠 占쎈┷占쎈선 �눧�똻�뻻占쎈쭢占쎈빍占쎈뼄.
            masked_error = pixel_error * background_mask

            # 5. 獄쏄퀗瑗� 占쎈동占쏙옙占쏙옙�벥 揶쏆뮇�땾嚥∽옙 占쎄돌占쎈듇占쎈선 占쎈즸域뱄옙 占쎌궎筌△뫀占쏙옙 ��④쑴沅쏉옙釉���⑨옙, 揶쏉옙餓λ쵐�뒄�몴占� ��④퉲鍮�占쎈빍占쎈뼄.
            # (�겫袁ⓦ걟占쎈퓠 占쎌삂占쏙옙占� 揶쏉옙(1e-8)占쎌뱽 占쎈쐭占쎈퉸 0占쎌몵嚥∽옙 占쎄돌占쎈듇占쎈뮉 野껉퍔�뱽 獄쎻뫗占쏙옙占쎈��占쎈빍占쎈뼄.)
            loss_mask = masked_error.sum() / (background_mask.sum() + 1e-8)
            loss_mask *= cfg.regularizer.lambda_mask

            # 6. 占쎌읈筌ｏ옙 占쎈��占쎈뼄占쎈퓠 占쎈쐭占쎈릭��⑨옙 嚥≪뮄�돪占쎈��占쎈빍占쎈뼄.
            loss += loss_mask
            loss_dict["mask"] = loss_mask
        '''
        # depthanythingv2 loss
        if need_depthanythingv2:
            loss_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 * depthanythingv2_loss(cam, render_pkg, iteration)
            loss += loss_depthanythingv2
            loss_dict["depthanythingv2"] = loss_depthanythingv2
        # if need_mast3r_metric_depth: #�뜝�럥�돯占쎈�롦뉩關援�???�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� mast3r_metric_depth ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
        #     loss += cfg.regularizer.lambda_mast3r_metric_depth * mast3r_metric_depth_loss(cam, render_pkg, iteration)

        if cfg.regularizer.lambda_ssim: #2. SSIM loss
            loss_ssim = cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
            loss += loss_ssim
            loss_dict["ssim"] = loss_ssim
        if cfg.regularizer.lambda_T_concen: #3. concentration Transmittance loss
            loss_T_concen = cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg[f'raw_T'])
            loss += loss_T_concen
            loss_dict["T_concen"] = loss_T_concen
        if cfg.regularizer.lambda_T_inside: #to encourage T to be inside the bounding box
            loss_T_inside = cfg.regularizer.lambda_T_inside * render_pkg[f'raw_T'].square().mean()
            loss += loss_T_inside
            loss_dict["T_inside"] = loss_T_inside
        if need_nd_loss: #mesh normal dmean loss(�뜝�럥夷ⓨ뜝�럥肉ョ뵳占� depth vs render_pkg['normal']->ray 占쎈쎗占쎈젻泳��떑�젂�뜝占� weight rendering)
            loss_nd_loss = cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
            loss += loss_nd_loss
            loss_dict["nd_loss"] = loss_nd_loss
        if need_nmed_loss: #mesh normal dmed loss(median depth vs render_pkg['normal']->ray 占쎈쎗占쎈젻泳��떑�젂�뜝占� weight rendering)
            loss_nmed_loss = cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)
            loss += loss_nmed_loss
            loss_dict["nmed_loss"] = loss_nmed_loss
        if need_pi3_normal_loss: #pi3 normal loss(render_pkg['normal']->ray 占쎈쎗占쎈젻泳��떑�젂�뜝占� weight rendering)
            lambda_pi3_mult = cfg.regularizer.pi3_normal_decay_mult ** (iteration // cfg.regularizer.pi3_normal_decay_every)
            loss_pi3_normal = cfg.regularizer.lambda_pi3_normal *lambda_pi3_mult* pi3_normal_loss(cam, render_pkg, iteration)
            loss += loss_pi3_normal
            loss_dict["pi3_normal_loss"] = loss_pi3_normal
        # lambda_R_concen loss? render_opt? 濚욌꼬�눊�꽠占쎈쑏�뜝占�? (�뜝�룞�삕占쎌돵占쎈㎥占쎈첓 ray 占쎌녃域뱄퐢荑덂뜝�럩援�?? �뜝�룞�삕占쎌돵占쎈㎥占쎈첓 癲ル슓堉곁땟怨살삕��억옙 l2 loss)
        # lambda_dist loss? render_opt? 濚욌꼬�눊�꽠占쎈쑏�뜝占�? (ray 占쎈쎗占쎈젻泳��떑�젂�뜝占�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援뀐┼�슢�뵯占쎌맄占쎈쨨�뜝占� density�뜝�럥�맶占쎈쐻�뜝占�?? 癲ル슢�뀈泳�怨�援℡뜝占�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援꿨뜝�럥�맶占쎈쐻�뜝占�??)
        # lambda_ascending loss? render_opt? 濚욌꼬�눊�꽠占쎈쑏�뜝占�?
        # lambda tv loss? voxel_model.optimizer.step() 濚욌꼬�눊�꽠占쎈쑏�뜝占�? (占쎌녃域뱄퐢荑덂뜝�럩援�?? 占쎈쨬占쎈즲占쎈쳮�뜝�럥爰� smoothness)
        # Backward to get gradient of current iteration
        
        if iteration % 100 == 0:  # 50 iter筌띾뜄�뼄 �빊�뮆�젾
            print(f"[iter {iteration}] loss breakdown:")
            for name, val in loss_dict.items():
                v = val.item()
                isn = torch.isnan(val)
                print(f"   {name:15s}: {v:.6e}{'  <-- NaN !!' if isn else ''}")

        # 占쎌읈筌ｏ옙 loss NaN 筌ｋ똾寃�
        if torch.isnan(loss):
            print(f"[iter {iteration}] 占쎌뒔�닼占� NaN detected in TOTAL loss!")
            for name, val in loss_dict.items():
                if torch.isnan(val):
                    print(f"   -> NaN found in {name}_loss")
        voxel_model.optimizer.zero_grad(set_to_none=True)

        



        loss.backward()
        '''
        if iteration % 100 == 0:
            g = voxel_model._geo_grid_pts.grad
            print(f"[iter {iteration}] geo_grad_norm = {g.norm().item():.4e}")
            if torch.isnan(g).any() or torch.isinf(g).any():
                bad_idx = torch.where(torch.isnan(g) | torch.isinf(g))
                print(f"   占쎌뒔�닼占� NaN/Inf in grad at indices: {bad_idx}")
                g_no_nan = g[~torch.isnan(g)]
                g_no_nan = g_no_nan[~torch.isinf(g_no_nan)]
                if g_no_nan.numel() > 0:
                    g_min, g_max = g_no_nan.min().item(), g_no_nan.max().item()
                else:
                    g_min, g_max = float('nan'), float('nan')
                print(f"   grad min/max: {g_min:.4e}/{g_max:.4e}")

        if iteration % 100 == 0:
            gnorm = voxel_model._geo_grid_pts.grad.norm().item()
            print(f"[iter {iteration}] geo_grad_norm = {gnorm:.4e}")
        '''
        if iteration % 100 == 0:
            with torch.no_grad():
                # non-leaf 마스크
                nonleaf_mask = (~voxel_model.is_leaf.view(-1).bool())

                # 레벨 벡터
                levels = voxel_model.octlevel.view(-1).to(torch.int64)

                # non-leaf들만 레벨별 개수 집계
                levels_nonleaf = levels[nonleaf_mask]
                if levels_nonleaf.numel() == 0:
                    print("  [voxels] non-leaf: 0")
                else:
                    uniq_lvls, counts = torch.unique(levels_nonleaf, return_counts=True)
                    order = torch.argsort(uniq_lvls)
                    uniq_lvls = uniq_lvls[order].tolist()
                    counts = counts[order].tolist()

                    total_nonleaf = int(sum(counts))
                    print(f"  [voxels] non-leaf total: {total_nonleaf}")
                    for L, C in zip(uniq_lvls, counts):
                        print(f"    - level {L-cfg.model.outside_level}: {C}")
        # INSERT_YOUR_CODE
        # 占쎌삏占쎈쐭筌랃옙 png 占쎌뵠占쎄숲占쎌쟿占쎌뵠占쎈�∽쭕�뜄�뼄 占쏙옙占쏙옙�삢
        '''
        import os
        import torchvision

        save_dir = "./output/ficus4/iter_debug_png"
        os.makedirs(save_dir, exist_ok=True)
        # render_image: [C, H, W] or [H, W, C] or [B, C, H, W]
        # Convert to [H, W, C] and clamp to [0,1] if needed
        img = render_image
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in [1,3]:  # [C,H,W]
                img = img.detach().cpu()
                img = img.clamp(0,1)
                img = torchvision.utils.make_grid(img)
                img = img.permute(1,2,0)  # [H,W,C]
            elif img.dim() == 3 and img.shape[-1] in [1,3]:  # [H,W,C]
                img = img.detach().cpu()
                img = img.clamp(0,1)
            elif img.dim() == 4:  # [B,C,H,W]
                img = img[0].detach().cpu()
                img = img.clamp(0,1)
                img = torchvision.utils.make_grid(img)
                img = img.permute(1,2,0)
            else:
                img = img.detach().cpu()
                img = img.clamp(0,1)
        # Convert to numpy and save
        import numpy as np
        from PIL import Image
        img_np = img.numpy()
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        img_np = (img_np * 255).astype(np.uint8)
        save_path = os.path.join(save_dir, f"iter_{iteration:06d}.png")
        Image.fromarray(img_np).save(save_path)
        '''
        # Grid-level regularization
        grid_reg_interval = iteration >= cfg.regularizer.tv_from and iteration <= cfg.regularizer.tv_until
        if cfg.regularizer.lambda_tv_density and grid_reg_interval:
            asdf = voxel_model._geo_grid_pts.grad * 10000
            lambda_tv_mult = cfg.regularizer.tv_decay_mult ** (iteration // cfg.regularizer.tv_decay_every)
            svraster_cuda.grid_loss_bw.total_variation(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                weight=cfg.regularizer.lambda_tv_density * lambda_tv_mult,
                vox_size_inv=voxel_model.vox_size_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.tv_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)
            if iteration % 100 == 0:
                diff = qwer - asdf
                # 占쎌젫��④퉲猷딀뉩占� (root mean square)
                rms = torch.sqrt(torch.mean(diff ** 2)).item()

                print("---")
                print(
                    f"min: {torch.min(diff).item():.6e}",   # 筌ㅼ뮇�꺖揶쏉옙 (��⑥눛釉곤옙�읅 占쎈ご疫뀐옙)
                    f"max: {torch.max(diff).item():.6e}",   # 筌ㅼ뮆占쏙옙揶쏉옙
                    f"rms: {rms:.6e}"                       # 占쎌젫��④퉲猷딀뉩占�
                )
        voxel_gradient_interval = iteration >= cfg.regularizer.vg_from and iteration <= cfg.regularizer.vg_until
        if cfg.regularizer.lambda_vg_density and voxel_gradient_interval:
            asdf = voxel_model._geo_grid_pts.grad * 10000
            G = voxel_model.vox_size_inv.numel()
            K =  int(G * (1.0 - float(cfg.regularizer.vg_drop_ratio)))
            active_list = torch.randperm(G, device=voxel_model.vox_key.device)[:K].to(torch.int32).contiguous()
            lambda_vg_mult = cfg.regularizer.vg_decay_mult ** (iteration // cfg.regularizer.vg_decay_every) * (G / K)
            svraster_cuda.grid_loss_bw.voxel_gradient(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                vox_size_inv = voxel_model.vox_size_inv,
                active_list=active_list,
                weight=cfg.regularizer.lambda_vg_density * lambda_vg_mult,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.vg_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)
            qwer = voxel_model._geo_grid_pts.grad * 10000
            if iteration % 100 == 0:
                diff = qwer - asdf
                # 占쎌젫��④퉲猷딀뉩占� (root mean square)
                rms = torch.sqrt(torch.mean(diff ** 2)).item()

                print("---")
                print(
                    f"min: {torch.min(diff).item():.6e}",   # 筌ㅼ뮇�꺖揶쏉옙 (��⑥눛釉곤옙�읅 占쎈ご疫뀐옙)
                    f"max: {torch.max(diff).item():.6e}",   # 筌ㅼ뮆占쏙옙揶쏉옙
                    f"rms: {rms:.6e}"                       # 占쎌젫��④퉲猷딀뉩占�
                )
        
        grid_eikonal_interval = iteration >= cfg.regularizer.ge_from and iteration <= cfg.regularizer.ge_until
        if cfg.regularizer.lambda_ge_density and grid_eikonal_interval:
            # if iteration == 3000:
            #     print("Eikonal loss applied (before)")
            asdf = voxel_model._geo_grid_pts.grad * 10000
            # breakpoint()
            lambda_ge_mult = cfg.regularizer.ge_decay_mult ** min(iteration // cfg.regularizer.ge_decay_every, 2)
            G = voxel_model.grid_keys.numel()
            K =  int(G * (1.0 - float(cfg.regularizer.ls_drop_ratio)))
            active_list = torch.randperm(G, device=voxel_model.grid_keys.device)[:K].to(torch.int32).contiguous()
            max_voxel_level = min(voxel_model.octlevel.max().item()-cfg.model.outside_level, 9)
            vox_size_min_inv = 2**max_voxel_level / voxel_model.inside_extent
            svraster_cuda.grid_loss_bw.grid_eikonal(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                grid_voxel_coord=grid_voxel_coord,
                grid_voxel_size=grid_voxel_size.view(-1),
                grid_res= 2**max_voxel_level,
                grid_mask=voxel_model.grid_mask,
                grid_keys= voxel_model.grid_keys,
                grid2voxel=voxel_model.grid2voxel,
                active_list=active_list,
                weight=cfg.regularizer.lambda_ge_density * lambda_ge_mult * (G / K),
                vox_size_inv=vox_size_min_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.ge_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)
            # if iteration == 1000:
            #     print("Eikonal loss applied (after)")
            qwer = voxel_model._geo_grid_pts.grad * 10000
            if iteration % 100 == 0:
                diff = qwer - asdf
                # 占쎌젫��④퉲猷딀뉩占� (root mean square)
                rms = torch.sqrt(torch.mean(diff ** 2)).item()

                print("---")
                print(
                    f"min: {torch.min(diff).item():.6e}",   # 筌ㅼ뮇�꺖揶쏉옙 (��⑥눛釉곤옙�읅 占쎈ご疫뀐옙)
                    f"max: {torch.max(diff).item():.6e}",   # 筌ㅼ뮆占쏙옙揶쏉옙
                    f"rms: {rms:.6e}"                       # 占쎌젫��④퉲猷딀뉩占�
                )
            # breakpoint()
        
        laplacian_interval = iteration >= cfg.regularizer.ls_from and iteration <= cfg.regularizer.ls_until
        if cfg.regularizer.lambda_ls_density and laplacian_interval:
            asdf = voxel_model._geo_grid_pts.grad * 10000
            lambda_ls_mult = cfg.regularizer.ls_decay_mult ** min(iteration // cfg.regularizer.ls_decay_every, 2)
            G = voxel_model.grid_keys.numel()
            K =  int(G * (1.0 - float(cfg.regularizer.ls_drop_ratio)))
            active_list = torch.randperm(G, device=voxel_model.grid_keys.device)[:K].to(torch.int32).contiguous()
            max_voxel_level = min(voxel_model.octlevel.max().item()-cfg.model.outside_level, 9)
            vox_size_min_inv = 2**max_voxel_level / voxel_model.inside_extent
            svraster_cuda.grid_loss_bw.laplacian_smoothness(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                grid_voxel_coord=grid_voxel_coord,
                grid_voxel_size=grid_voxel_size.view(-1),
                grid_res= 2**max_voxel_level,
                grid_mask=voxel_model.grid_mask,
                grid_keys= voxel_model.grid_keys,
                grid2voxel=voxel_model.grid2voxel,
                active_list=active_list,
                weight=cfg.regularizer.lambda_ls_density * lambda_ls_mult * (G / K),
                vox_size_inv=vox_size_min_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.ls_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)
            qwer = voxel_model._geo_grid_pts.grad * 10000
            if iteration % 100 == 0:
                diff = qwer - asdf
                # 占쎌젫��④퉲猷딀뉩占� (root mean square)
                rms = torch.sqrt(torch.mean(diff ** 2)).item()

                print("---")
                print(
                    f"min: {torch.min(diff).item():.6e}",   # 筌ㅼ뮇�꺖揶쏉옙 (��⑥눛釉곤옙�읅 占쎈ご疫뀐옙)
                    f"max: {torch.max(diff).item():.6e}",   # 筌ㅼ뮆占쏙옙揶쏉옙
                    f"rms: {rms:.6e}"                       # 占쎌젫��④퉲猷딀뉩占�
                )
            
        points_interval = iteration >= cfg.regularizer.points_loss_from and cfg.regularizer.points_loss_until >= iteration
        if cfg.regularizer.lambda_points_density and points_interval:
            # first sample points from points
            # Sample random points from initial_points according to points_sample_rate
            asdf = voxel_model._geo_grid_pts.grad * 10000
            sample_rate = cfg.regularizer.points_sample_rate
            num_points = initial_points.shape[0]
            num_sample = max(1, int(num_points * sample_rate))
            idx = torch.randperm(num_points, device=initial_points.device)[:num_sample]
            sampled_points = initial_points[idx]
            points_in_grid = (sampled_points - (voxel_model.scene_center - voxel_model.inside_extent*0.5)) / voxel_model.inside_extent* (2**max_voxel_level)
            # Sample exactly 100 points if possible, otherwise use all available points
            lambda_points_mult = cfg.regularizer.points_loss_decay_mult ** (iteration // cfg.regularizer.points_loss_decay_every)
            svraster_cuda.grid_loss_bw.points_loss(
                points_in_grid=points_in_grid,
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                grid_voxel_coord=grid_voxel_coord,
                grid_voxel_size=grid_voxel_size.view(-1),
                grid_res= 2**max_voxel_level,
                grid_mask=voxel_model.grid_mask,
                grid_keys= voxel_model.grid_keys,
                grid2voxel=voxel_model.grid2voxel,
                weight=cfg.regularizer.lambda_points_density * lambda_points_mult,
                vox_size_inv=vox_size_min_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.points_loss_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad
            )
            qwer = voxel_model._geo_grid_pts.grad * 10000
            if iteration % 100 == 0:
                diff = qwer - asdf
                rms = torch.sqrt(torch.mean(diff ** 2)).item()
                print("---")
                print(
                    f"min: {torch.min(diff).item():.6e}",  
                    f"max: {torch.max(diff).item():.6e}",   
                    f"rms: {rms:.6e}"                       
                )
        
        # Optimizer step
        voxel_model.optimizer.step()  # SVOptimizer
        
        # Learning rate warmup scheduler step
        if iteration <= cfg.optimizer.n_warmup: #learning rate warmup
            rate = iteration / cfg.optimizer.n_warmup
            for param_group in voxel_model.optimizer.param_groups:
                param_group["lr"] = rate * param_group["base_lr"]
        
        
        for pg in voxel_model.optimizer.param_groups:
            if pg.get("name") == "_geo_grid_pts":
                if iteration < 100:
                    pg["lr"] = 0.0
                    pg["base_lr"] = 0.0  # warmup이 다시 키우지 못하게 base도 0으로
                elif iteration == 100:
                    val = cfg.optimizer.geo_lr  # 위에서 세팅해 둔 최종 geo lr
                    pg["lr"] = val
                    pg["base_lr"] = val
        
        

        if iteration in cfg.optimizer.lr_decay_ckpt: #learning rate decay
            for param_group in voxel_model.optimizer.param_groups:
                ori_lr = param_group["lr"]
                param_group["lr"] *= cfg.optimizer.lr_decay_mult
                print(f'LR decay of {param_group["name"]}: {ori_lr} => {param_group["lr"]}')
            cfg.regularizer.lambda_vg_density *= cfg.optimizer.lr_decay_mult
            cfg.regularizer.lambda_tv_density *= cfg.optimizer.lr_decay_mult
            cfg.regularizer.lambda_ge_density *= cfg.optimizer.lr_decay_mult
            cfg.regularizer.lambda_ls_density *= cfg.optimizer.lr_decay_mult
            #cfg.regularizer._ge_final *= cfg.optimizer.lr_decay_mult
            #cfg.regularizer._ls_final *= cfg.optimizer.lr_decay_mult

        #decay just geo_lr
        
        '''
        if iteration == 2000 or iteration == 4000 or iteration ==6000:
            cfg.optimizer.geo_lr *=0.5
            cfg.regularizer.lambda_ge_density *= 0.5
            cfg.regularizer.lambda_ls_density *= 0.5
            cfg.regularizer._ge_final *= 0.5
            cfg.regularizer._ls_final *= 0.5
            '''

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
            # �뜝�럥�맶占쎈쐻�뜝占�?? voxel癲ル슢�뵯占쎌맄�뜝�럥�렡 gradient�뜝�럥�맶占쎈쐻�뜝占�?? 癲ル슢�뀈泳�怨살삕筌뤿벝�삕占쎌맶占쎈쐻�뜝占�?? ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� -> subdivide ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�

        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        need_pruning = ( #prune_every = 1000
            iteration % cfg.procedure.prune_every == 0 and \
            iteration >= cfg.procedure.prune_from and \
            iteration <= cfg.procedure.prune_until)
        if iteration == 1:
            if cfg.procedure.prune_from ==0:
                need_pruning = True
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
        # max_w = stat_pkg['max_w'] # 占쎈쨬占쎈즸占쎌굲 voxel癲ル슢�뵯占쎌맄�뜝�럥�렡 max weight (T_i*a_i)
        # min_samp_interval = stat_pkg['min_samp_interval'] # 占쎈쨬占쎈즸占쎌굲 voxel癲ル슢�뵯占쎌맄�뜝�럥�렡 min sampling interval (T_i*a_i)
        # view_cnt = stat_pkg['view_cnt'] # 占쎈쨬占쎈즸占쎌굲 voxel癲ル슢�뵯占쎌맄�뜝�럥�렡 view count
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
            if cfg.model.density_mode == 'sdf'  and iteration >=1000:
                sdf_vals = voxel_model._geo_grid_pts[voxel_model.vox_key]  # [N, 8, 1]
                signs = (sdf_vals > 0).float()
                has_surface = (signs.min(dim=1).values != signs.max(dim=1).values).view(-1)

                min_abs_sdf = sdf_vals.abs().min(dim=1).values.view(-1)  # avg??
                global_vox_size_min = voxel_model.vox_size.min().item()
                #sdf_thresh = learning_thickness*2 * global_vox_size_min
                sdf_thresh = torch.log(torch.tensor(199.0, device=voxel_model._log_s.device)) / torch.exp(10 * voxel_model._log_s)
                learning_thickness = sdf_thresh /2/global_vox_size_min
                print(f"true_learning_thickness = {learning_thickness:.4f}")
                sdf_thresh = max(2*global_vox_size_min, sdf_thresh.item())
                print(f"augmented_learning_thickness = {sdf_thresh/global_vox_size_min/2:.4f}")
                # sdf_thresh 占쎈쨬占쎈즴�씙�뜝�럡�떖: 3000 -> 15000 -> 0.5 -> 0.2
                '''
                start_iter = 3000
                end_iter = 15000
                start_thresh = 0.5
                end_thresh = 0.2
                progress = min(1.0, max(0.0, (iteration - start_iter) / (end_iter - start_iter)))
                sdf_thresh = start_thresh + (end_thresh - start_thresh) * progress
                ''''''
                if iteration ==2000:
                    sdf_thresh *=2
                    '''
                prune_mask = (~has_surface) & (min_abs_sdf > sdf_thresh) 
            elif cfg.model.density_mode == 'sdf':
                sdf_vals = voxel_model._geo_grid_pts[voxel_model.vox_key]  # [N, 8, 1]
                min_abs_sdf = sdf_vals.abs().min(dim=1).values.view(-1)  # avg??
                global_vox_size_min = voxel_model.vox_size.min().item()
                #sdf_thresh = learning_thickness*2 * global_vox_size_min
                sdf_thresh = torch.log(torch.tensor(199.0, device=voxel_model._log_s.device)) / torch.exp(10 * voxel_model._log_s)
                learning_thickness = sdf_thresh /2/global_vox_size_min
                print(f"true_learning_thickness = {learning_thickness:.4f}")
                sdf_thresh = max(2*global_vox_size_min, sdf_thresh.item())
                print(f"augmented_learning_thickness = {sdf_thresh/global_vox_size_min/2:.4f}")
                # sdf_thresh 占쎈쨬占쎈즴�씙�뜝�럡�떖: 3000 -> 15000 -> 0.5 -> 0.2
                
                prune_mask =  (min_abs_sdf > sdf_thresh) 
            voxel_model.pruning(prune_mask)

            # Prune statistic (for the following subdivision)
            kept_idx = (~prune_mask).argwhere().squeeze(1)
            for k, v in stat_pkg.items():
                stat_pkg[k] = v[kept_idx] #v?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� [N, 1]癲ル슣�돵獒뺣냵�삕�걡占� tensor, pruning ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�

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
                non_finest_mask = voxel_model.octlevel.squeeze(1) < (svraster_cuda.meta.MAX_NUM_LEVELS-5- max(0, 3 - iteration // 2000)+cfg.model.outside_level)
                print(f"max octlevel for sdf: {svraster_cuda.meta.MAX_NUM_LEVELS-2- max(0, 3 - iteration // 3000)}")
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
            outside_mask = ~voxel_model.inside_mask
            subdivide_mask_for_outside = subdivide_mask & outside_mask
            n_out = int(subdivide_mask_for_outside.sum().item())
            print(f"[outside subdivide] count = {n_out}")

            
            inside = voxel_model.inside_mask
            candidates = valid_mask & inside       # 내부 후보만

            if iteration <= cfg.procedure.subdivide_all_until:
                subdivide_mask = candidates
            else:
                prio = voxel_model.subdiv_meta.squeeze(1)

                # inside 후보들에서만 분위값 계산
                prio_inside = prio[candidates]
                if prio_inside.numel() == 0:
                    subdivide_mask = torch.zeros_like(candidates)
                else:
                    thres = torch.quantile(prio_inside, 1 - subdivide_prop)
                    subdivide_mask = candidates & (prio >= thres) | subdivide_mask_for_outside
            if cfg.model.density_mode == 'sdf' and iteration <6000: # SDF 癲ル슢�뀈泳�占썹뛾占썲뜝占�?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� ?�뜝�럥�맶�뜝�럥吏쀥뜝�럩援�
                with torch.no_grad():
                    sdf_vals = voxel_model._geo_grid_pts[voxel_model.vox_key]  # [N, 8, 1]
                    signs = (sdf_vals > 0).float()
                    has_surface = (signs.min(dim=1).values != signs.max(dim=1).values).view(-1)

                cur_level = voxel_model.octlevel.squeeze(1)
                max_level = 9 +cfg.model.outside_level- max(0, 2 - iteration // 2000) # 9 + 1 = 10
                under_level = cur_level < max_level
                valid_mask = has_surface & under_level
                subdivide_mask = (valid_mask & voxel_model.is_leaf.squeeze(1) & voxel_model.inside_mask) | ( subdivide_mask_for_outside &valid_mask)
                '''
                if iteration < 6000:
                    topk = 50000

                    valid_indices = torch.where(valid_mask)[0]
                    
                    valid_priorities = priority[valid_mask]

                    k = min(topk, valid_priorities.numel())
                    topk_rel_idx = torch.topk(valid_priorities, k).indices
                    topk_abs_idx = valid_indices[topk_rel_idx]

                    subdivide_mask = torch.zeros_like(priority, dtype=torch.bool)
                    subdivide_mask[topk_abs_idx] = True
                '''
                if  iteration <= cfg.procedure.subdivide_all_until:
                    subdivide_mask = under_level
            
            print(f"voxel_model._log_s = {voxel_model._log_s}")
            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (rank > rank[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            #subdivide_mask = (subdivide_mask & voxel_model.is_leaf.squeeze(1) & voxel_model.inside_mask) | subdivide_mask_for_outside
            
            ori_n = voxel_model.num_voxels
            if subdivide_mask.sum() > 0:
                voxel_model.subdividing(subdivide_mask, cfg.procedure.subdivide_save_gpu)
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')
            max_voxel_level = voxel_model.octlevel.max().item()-cfg.model.outside_level
            print(f'level max : {max_voxel_level}') 
            voxel_model.subdiv_meta.zero_()  # reset subdiv meta
            remain_subdiv_times -= 1
            torch.cuda.empty_cache()
        if (need_pruning or need_subdividing) :
            #vox_size_min_inv = 1.0 / voxel_model.vox_size.min().item() 
            #print(f"voxel_model.vox_size_min_inv = {vox_size_min_inv:.4f}")
            max_voxel_level = min(voxel_model.octlevel.max().item()-cfg.model.outside_level, 9)
            grid_voxel_coord = ((voxel_model.vox_center- voxel_model.vox_size * 0.5)-(voxel_model.scene_center-voxel_model.inside_extent*0.5))/voxel_model.inside_extent*(2**max_voxel_level) #占쎌녃域뱄퐢荑덂뜝�럩援�???�뜝�럥�맶�뜝�럥吏쀥뜝�럩援� �뜝�럥苑욃뜝�럩�뮡嶺뚮쪋�삕
            grid_voxel_coord = torch.round(grid_voxel_coord).float()
            grid_voxel_size = (voxel_model.vox_size / voxel_model.inside_extent) * (2**max_voxel_level) 
            grid_voxel_size = torch.round(grid_voxel_size).float()
            #print(f"grid_voxel_coord = {grid_voxel_coord}")
            #print(f"grid_voxel_size = {grid_voxel_size}") 
            
            #print(f"voxel_model.vox_size_min_inv = {vox_size_min_inv:.4f}") 

            voxel_model.grid_mask, voxel_model.grid_keys, voxel_model.grid2voxel = octree_utils.update_valid_gradient_table(cfg.model.density_mode, voxel_model.vox_center, voxel_model.vox_size, voxel_model.scene_center, voxel_model.inside_extent, max_voxel_level, voxel_model.is_leaf)
            torch.cuda.synchronize()

            '''
            print(f"voxel_model.grid_mask = {voxel_model.grid_mask}") #grid_mask �뜝�럥�돯�뜝�럥痢듿뜝�럩議�
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
                im_tensor2np(-view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(-view.depth2normal(render_depth_med) * 0.5 + 0.5),
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
        cfg.optimizer.geo_lr = 0.005
        cfg.optimizer.sh0_lr = 0.01 #0.01
        cfg.optimizer.shs_lr = 0.00025 # 0.00025
        cfg.optimizer.lr_decay_ckpt = [ 4000,6000,7000]
        cfg.optimizer.lr_decay_mult = 0.5
        cfg.init.geo_init = 0.0
        cfg.regularizer.dist_from = 4000
        cfg.regularizer.lambda_dist = 0.0
        cfg.regularizer.lambda_tv_density = 0.0
        cfg.regularizer.tv_from = 0000
        cfg.regularizer.tv_until = 4000
        cfg.regularizer.lambda_ascending = 0.0
        cfg.regularizer.ascending_from = 0
        cfg.regularizer.lambda_normal_dmean = 0.001
        cfg.regularizer.n_dmean_from = 2000  # 
        cfg.regularizer.lambda_normal_dmed = 0.001
        cfg.regularizer.n_dmed_from = 1000
        cfg.procedure.prune_from = 00
        cfg.procedure.prune_every = 1000
        cfg.procedure.prune_until = 9000
        cfg.procedure.subdivide_from = 0
        cfg.procedure.subdivide_every = 250
        cfg.regularizer.lambda_mask = 0.0

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
            for v in cfg.procedure.reset_sh_ckpt] #scale 占쎈뙑��⑤슦占쏙퐦�삕占쎌젳

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
