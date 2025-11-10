# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import argparse
from yacs.config import CfgNode


cfg = CfgNode()

cfg.model = CfgNode(dict(
    vox_geo_mode = "triinterp1",
    density_mode = "exp_linear_11",
    sh_degree = 3,
    ss = 1.5,
    outside_level = 5,  # Number of Octree level outside the main 3D region
    model_path = "",
    white_background = False,  # Assume known white bg color
    black_background = False,  # Assume known black bg color
))

cfg.data = CfgNode(dict(
    source_path = "",
    images = "images",
    res_downscale = 0.,
    res_width = 0,
    extension = ".png",
    blend_mask = True,
    depth_paths = "",
    normal_paths = "/home/user/data/TNT_GOF/TrainingSet/Barn/normal/*.png",
    depth_scale = 1.0,
    data_device = "cpu",
    eval = False,
    test_every = 8, # Only if dataset has no testset && eval=True
))

cfg.bounding = CfgNode(dict(
    # Define the main (inside) region bounding box
    # The default use the suggested bounding if given by dataset.
    # Otherwise, it automatically chose from forward or camera_median modes.
    # See src/utils/bounding_utils.py for details.

    # default | camera_median | camera_max | forward | pcd
    bound_mode = "default", # The main region bounding mode
    bound_scale = 1.0,        # Scaling factor of the bound
    forward_dist_scale = 1.0, # For forward mode
    pcd_density_rate = 0.1,   # For pcd mode
))

cfg.optimizer = CfgNode(dict(
    geo_lr = 0.025,
    sh0_lr = 0.010,
    shs_lr = 0.00025,
    log_s_lr = 0.001, # osh ?���??????? ?��?��?��?��

    optim_beta1 = 0.1,
    optim_beta2 = 0.99,
    optim_eps = 1e-15,

    n_warmup = 100,

    lr_decay_ckpt = [19000],
    lr_decay_mult = 0.1,
))

cfg.regularizer = CfgNode(dict(
    # Main photometric loss
    lambda_photo = 1.0,
    use_l1 = False,
    use_huber = False,
    huber_thres = 0.03,

    # SSIM loss
    lambda_ssim = 0.02,

    # Sparse depth loss
    lambda_sparse_depth = 0.0,
    sparse_depth_until = 10_000,

    # Mask loss
    lambda_mask = 0.0,

    # Depthanything loss
    lambda_depthanythingv2 = 0.0,
    depthanythingv2_from = 00,
    depthanythingv2_end = 8000,
    depthanythingv2_end_mult = 0.1,

    # Mast3r metrid loss
    lambda_mast3r_metric_depth = 0.0,
    mast3r_repo_path = '',
    mast3r_metric_depth_from = 3000,
    mast3r_metric_depth_end = 20000,
    mast3r_metric_depth_end_mult = 0.01,

    # Final transmittance should concentrate to either 0 or 1
    lambda_T_concen = 0.0,

    # Final transmittance should be 0
    lambda_T_inside = 0.0,

    # Per-point rgb loss
    lambda_R_concen = 0.01, #0.01

    # Geometric regularization
    lambda_ascending = 0.0,
    ascending_from = 0,
    ascending_until = 2000,

    # Distortion loss (encourage distribution concentration on ray)
    lambda_dist = 0.1,
    dist_from = 10000,

    # Consistency loss of rendered normal and derived normal from expected depth
    lambda_normal_dmean = 0.0,
    n_dmean_from = 2_000,
    n_dmean_end = 20_000,
    n_dmean_ks = 3,
    n_dmean_tol_deg = 90.0,

    # Consistency loss of rendered normal and derived normal from median depth
    lambda_normal_dmed = 0.0,
    n_dmed_from=2000,
    n_dmed_end=20_000,

    lambda_pi3_normal = 0.1,
    pi3_normal_from=0,
    pi3_normal_end=4000,
    pi3_normal_decay_every=2000,
    pi3_normal_decay_mult=1.0,

    # Total variation loss of density grid
    lambda_tv_density = 1e-8,
    tv_from = 0,
    tv_until = 10000,
    tv_decay_every = 1000,
    tv_decay_mult = 0.8,
    tv_sparse = False,

    lambda_vg_density = 1e-11,
    vg_from = 6000,
    vg_until = 8000,
    vg_decay_every = 2000,
    vg_decay_mult = 0.25,
    vg_sparse = False,
    vg_drop_ratio = 0.5,

    lambda_ge_density = 2e-8,
    ge_from = 0,
    ge_until = 6000,
    ge_decay_every = 2000,
    ge_decay_mult = 0.25,
    ge_sparse = False,
    ge_drop_ratio = 0.0,

    lambda_ls_density = 1e-10,
    ls_from = 0,
    ls_until = 8000,
    ls_decay_every = 2000,
    ls_decay_mult = 0.25,
    ls_sparse = False,
    ls_drop_ratio = 0.0,

    lambda_points_density = 0.00000,
    points_loss_from = 0,
    points_loss_until = 4000,
    points_loss_decay_every = 2000,
    points_loss_decay_mult = 1.0,
    points_loss_sparse = False,
    points_sample_rate = 0.05,
    # Data augmentation
    ss_aug_max = 1,
    rand_bg = False,

))

cfg.init = CfgNode(dict(
    # Voxel property initialization
    geo_init = -10.0, 
    sh0_init = 0.5,
    shs_init = 0.0,
    log_s_init = 0.3, #?��?��?�� ?��?�� ?��?�� ?��?��

    sh_degree_init = 3,

    # Init main inside region by dense voxels
    init_n_level = 6,  # (2^6)^3 voxels

    # Init background outside region by different strategies.
    #   none        : no voxels in outside region.
    #   uniform[N]  : subdivide each shell level by N times.
    #                 ex. uniform1, uniform2, uniform3, ...
    #   heuristic   : init fixed amount of voxels based on init_out_ratio.
    outside_mode = "heuristic",
    init_out_ratio = 2.0,
    
    # Apply aabb crop if given
    aabb_crop = False,

    init_sparse_points = True,
))

cfg.procedure = CfgNode(dict(
    # Schedule
    n_iter = 10_000,
    sche_mult = 1.0,
    seed=3721,

    # Reset sh
    reset_sh_ckpt = [-1],

    # Adaptive voxel pruning
    prune_from = 1000,
    prune_every = 1000,
    prune_until = 18000,
    prune_thres_init = 0.0001,
    prune_thres_final = 0.03,

    # Adaptive voxel pruning
    subdivide_from = 1000,
    subdivide_every = 1000,
    subdivide_until = 9000,
    subdivide_samp_thres = 1.0, # A voxel max sampling rate should larger than this.
    subdivide_target_scale = 90.0,
    subdivide_max_num = 10_000_000,
    subdivide_all_until = 0,
    subdivide_save_gpu = False,
))

cfg.auto_exposure = CfgNode(dict(
    enable = False,
    auto_exposure_upd_ckpt = [5000, 10000, 15000]
))

for i_cfg in cfg.values():
    i_cfg.set_new_allowed(True)


def everytype2bool(v):
    if v.isnumeric():
        return bool(int(v))
    v = v.lower()
    if v in ['n', 'no', 'none', 'false']:
        return False
    return True


def update_argparser(parser):
    for name in cfg.keys():
        group = parser.add_argument_group(name)
        for key, value in getattr(cfg, name).items():
            t = type(value)

            if t == bool:
                group.add_argument(f"--{key}", action='store_true' if t else 'store_false')
            elif t == list:
                group.add_argument(f"--{key}", default=value, type=type(value[0]), nargs="*")
            elif t == tuple:
                group.add_argument(f"--{key}", default=value, type=type(value[0]), nargs=len(value))
            else:
                group.add_argument(f"--{key}", default=value, type=t)


def update_config(cfg_files, cmd_lst=[]):
    # Update from config files
    if isinstance(cfg_files, str):
        cfg_files = [cfg_files]
    for cfg_path in cfg_files:
        cfg.merge_from_file(cfg_path)

    if len(cmd_lst) == 0:
        return

    # Parse the arguments from command line
    internal_parser = argparse.ArgumentParser()
    update_argparser(internal_parser)
    internal_args = internal_parser.parse_args(cmd_lst)

    # Update from command line args
    for name in cfg.keys():
        cfg_subgroup = getattr(cfg, name)
        for key in cfg_subgroup.keys():
            arg_val = getattr(internal_args, key)
            # Check if the default values is updated
            if internal_parser.get_default(key) != arg_val:
                cfg_subgroup[key] = arg_val


