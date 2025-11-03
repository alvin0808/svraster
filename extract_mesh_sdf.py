# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import math
import time
import numpy as np
from tqdm import tqdm
import trimesh

import torch
import svraster_cuda

from src.config import cfg, update_argparser, update_config
from src.utils import octree_utils
from src.utils import activation_utils
from src.sparse_voxel_gears.adaptive import subdivide_by_interp, agg_voxel_into_grid_pts

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel

from src.utils.fuser_utils import Fuser


def tsdf_fusion(
        cam_lst, depth_lst, alpha_lst,
        grid_pts_xyz, trunc_dist, crop_border, alpha_thres):

    assert len(cam_lst) == len(depth_lst)
    assert len(cam_lst) == len(alpha_lst)

    fuser = Fuser(
        xyz=grid_pts_xyz,
        bandwidth=trunc_dist,
        use_trunc=True,
        fuse_tsdf=True,
        feat_dim=0,
        alpha_thres=alpha_thres,
        crop_border=crop_border,
        normal_weight=False,
        depth_weight=False,
        border_weight=False,
        use_half=False)

    for cam, frame_depth, frame_alpha in zip(tqdm(cam_lst), depth_lst, alpha_lst):

        frame_depth = frame_depth.cuda()
        frame_alpha = frame_alpha.cuda()

        fuser.integrate(cam, frame_depth, alpha=frame_alpha)

    tsdf = fuser.tsdf.squeeze(1).contiguous()
    return tsdf


def extract_mesh_progressive(args, data_pack, voxel_model, init_lv, final_lv, crop_bbox):

    # Render depth and alphas
    cam_lst = data_pack.get_train_cameras()
    depth_lst = []
    alpha_lst = []
    for cam in tqdm(cam_lst, desc="Render training views"):
        render_pkg = voxel_model.render(cam, output_depth=True, output_T=True)
        if args.use_mean:
            frame_depth = render_pkg['raw_depth'][[0]]  # Use mean depth
        else:
            frame_depth = render_pkg['raw_depth'][[2]]  # Use median depth
        frame_alpha = 1 - render_pkg['raw_T']
        if args.save_gpu:
            frame_depth = frame_depth.cpu()
            frame_alpha = frame_alpha.cpu()
        depth_lst.append(frame_depth)
        alpha_lst.append(frame_alpha)

    # Determine bounding volume for marching cube
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")

    # Construct a initial dense grid
    octpath, octlevel = octree_utils.gen_octpath_dense(
        outside_level=voxel_model.outside_level,
        n_level_inside=init_lv)
    grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath, octlevel)
    grid_pts_xyz = octree_utils.compute_gridpoints_xyz(grid_pts_key, voxel_model.scene_center, voxel_model.scene_extent)

    # Filter outside
    grid_inside_mask = ((inside_min <= grid_pts_xyz) & (grid_pts_xyz <= inside_max)).all(-1)
    vox_inside_mask = grid_inside_mask[vox_key].any(-1)
    vox_inside_idx = torch.where(vox_inside_mask)[0]
    octpath = octpath[vox_inside_idx]
    octlevel = octlevel[vox_inside_idx]
    grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath, octlevel)
    grid_pts_xyz = octree_utils.compute_gridpoints_xyz(grid_pts_key, voxel_model.scene_center, voxel_model.scene_extent)

    # Run progressive TSDF fusion
    print(f'TSDF levels from {init_lv} to {final_lv}')
    for lv in range(init_lv, final_lv+1):

        # Determine trunction
        now_level = torch.tensor([voxel_model.outside_level + min(lv, args.trunc_lv)], device="cuda")
        now_voxel_size = octree_utils.level_2_vox_size(voxel_model.scene_extent, now_level).item()
        trunc_dist = args.trunc_vox * now_voxel_size

        print(f"Running lv={lv:2d}: #voxels={len(octpath)}; vox_size={now_voxel_size}; trunc={trunc_dist}")

        # Run tsdf fusion at current levels
        grid_tsdf = tsdf_fusion(
            cam_lst, depth_lst, alpha_lst,
            grid_pts_xyz, trunc_dist, args.crop_border, args.alpha_thres)

        # Merge from previous levels
        if lv < final_lv:
            # Remove some voxels
            vox_tsdf = grid_tsdf[vox_key]
            prune_mask = vox_tsdf.isnan().any(-1) | (vox_tsdf.amax(1) < -args.pg_prune) | (vox_tsdf.amin(1) > args.pg_prune)
            filter_idx = torch.where(~prune_mask)[0]
            octpath = octpath[filter_idx]
            octlevel = octlevel[filter_idx]

            # Subdivide voxels
            octpath, octlevel = octree_utils.gen_children(octpath, octlevel)
            grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath, octlevel)
            grid_pts_xyz = octree_utils.compute_gridpoints_xyz(grid_pts_key, voxel_model.scene_center, voxel_model.scene_extent)

            del grid_tsdf, vox_tsdf
            torch.cuda.empty_cache()

    verts, faces = svraster_cuda.marching_cubes.torch_marching_cubes_grid(
        grid_pts_val=grid_tsdf,
        grid_pts_xyz=grid_pts_xyz,
        vox_key=vox_key,
        iso=0)
    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh


def extract_mesh(args, data_pack, voxel_model, final_lv, crop_bbox, use_lv_avg, iso=0):

    # Render depth and alphas
    cam_lst = data_pack.get_train_cameras()
    depth_lst = []
    alpha_lst = []
    for cam in tqdm(cam_lst, desc="Render training views"):
        render_pkg = voxel_model.render(cam, output_depth=True, output_T=True)
        if args.use_mean:
            frame_depth = render_pkg['raw_depth'][[0]]  # Use mean depth
        else:
            frame_depth = render_pkg['raw_depth'][[2]]  # Use median depth
        frame_alpha = 1 - render_pkg['raw_T']
        if args.save_gpu:
            frame_depth = frame_depth.cpu()
            frame_alpha = frame_alpha.cpu()
        depth_lst.append(frame_depth)
        alpha_lst.append(frame_alpha)

    # Filter background voxels
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")
    inside_mask = ((inside_min <= voxel_model.grid_pts_xyz) & (voxel_model.grid_pts_xyz <= inside_max)).all(-1)
    inside_mask = inside_mask[voxel_model.vox_key].any(-1)
    inside_idx = torch.where(inside_mask)[0]

    octpath = voxel_model.octpath[inside_idx]
    octlevel = voxel_model.octlevel[inside_idx]

    # Clamp levels
    target_level = voxel_model.outside_level + final_lv
    octpath, octlevel = octree_utils.clamp_level(octpath, octlevel, target_level)
    print(f'Voxel levels from {octlevel.min()} to {octlevel.max()}')

    # Construct grid points
    grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath, octlevel)
    grid_pts_xyz = octree_utils.compute_gridpoints_xyz(grid_pts_key, voxel_model.scene_center, voxel_model.scene_extent)

    # Run tsdf fusion
    vox_level = torch.tensor([voxel_model.outside_level + args.trunc_lv], device="cuda")
    vox_size = octree_utils.level_2_vox_size(voxel_model.scene_extent, vox_level).item()
    trunc_dist = args.trunc_vox * vox_size
    print(f"Running adaptive: #voxels={len(octpath)} / finest vox_size={voxel_model.vox_size.min().item()} / trunc={trunc_dist}")
    grid_tsdf = tsdf_fusion(
        cam_lst, depth_lst, alpha_lst,
        grid_pts_xyz, trunc_dist, args.crop_border, args.alpha_thres)

    if use_lv_avg:
        while True:
            n_ori = len(octlevel)
            unit_val = grid_tsdf[vox_key]

            # Filter
            mask = (unit_val > iso).any(1) & (unit_val < iso).any(1) & ~unit_val.isnan().any(1)
            filter_idx = torch.where(mask)[0]
            octpath = octpath[filter_idx]
            octlevel = octlevel[filter_idx]
            unit_val = unit_val[filter_idx]

            # Compute children
            mask = (octlevel.squeeze() < target_level)
            kept_idx = torch.where(~mask)[0]
            subdiv_idx = torch.where(mask)[0]
            if len(subdiv_idx) == 0:
                break
            child_octpath, child_octlevel = octree_utils.gen_children(octpath[subdiv_idx], octlevel[subdiv_idx])
            child_unit_val = subdivide_by_interp(unit_val[subdiv_idx])

            # Compute new voxels and tsdf grid points
            octpath = torch.cat([octpath[kept_idx], child_octpath])
            octlevel = torch.cat([octlevel[kept_idx], child_octlevel])
            unit_val = torch.cat([unit_val[kept_idx], child_unit_val])
            grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath, octlevel)
            grid_pts_xyz = octree_utils.compute_gridpoints_xyz(grid_pts_key, voxel_model.scene_center, voxel_model.scene_extent)
            grid_tsdf = agg_voxel_into_grid_pts(len(grid_pts_xyz), vox_key, unit_val)

            n_new = len(octlevel)
            print(f"Subdiv {n_ori:10d} => {n_new:10d}")

        del unit_val, grid_pts_key, filter_idx, kept_idx, subdiv_idx

    del octpath, octlevel
    torch.cuda.empty_cache()

    verts, faces = svraster_cuda.marching_cubes.torch_marching_cubes_grid(
        grid_pts_val=grid_tsdf,
        grid_pts_xyz=grid_pts_xyz,
        vox_key=vox_key,
        iso=iso)
    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh

def extract_mesh_sdf(args, voxel_model, final_lv, crop_bbox, use_lv_avg, iso=0.0):
    import torch
    import trimesh
    from tqdm import tqdm

    # --------------------------
    # 1. Crop bounding box ?��?���??
    # --------------------------
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")
    inside_mask = ((inside_min <= voxel_model.grid_pts_xyz) & (voxel_model.grid_pts_xyz <= inside_max)).all(-1)
    inside_mask = inside_mask[voxel_model.vox_key].any(-1)
    inside_idx = torch.where(inside_mask)[0]

    # --------------------------
    # 2. Clamp voxel level
    # --------------------------
    octpath = voxel_model.octpath[inside_idx]
    octlevel = voxel_model.octlevel[inside_idx]
    target_level = voxel_model.outside_level + final_lv
    octpath, octlevel = octree_utils.clamp_level(octpath, octlevel, target_level)
    print(f'[SDF] Voxel levels from {octlevel.min()} to {octlevel.max()}')

    # --------------------------
    # 3. Grid 좌표 계산
    # --------------------------
    grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath, octlevel)
    #grid_pts_key -> n_grid개의 grid points?�� ????�� octree key
    #vox_key -> n_voxel개의 voxel�?? 8개의 grid points 
    grid_pts_xyz = octree_utils.compute_gridpoints_xyz(voxel_model.grid_pts_key, voxel_model.scene_center, voxel_model.scene_extent)
    
    # --------------------------
    # 4. SDF corner value 추출 ?��?�� ?��?��
    # --------------------------
    grid_sdf_vals = voxel_model._geo_grid_pts
    print(f'[SDF] Grid points: {len(grid_pts_xyz)}')
    # --------------------------
    # 5. SDF aggregation + optional subdivision
    # --------------------------
    if use_lv_avg:
        while True:
            n_ori = len(octlevel)
            unit_val = grid_sdf_vals[vox_key]

            # zero-crossing filter
            mask = (unit_val > iso).any(1) & (unit_val < iso).any(1) & ~unit_val.isnan().any(1)
            filter_idx = torch.where(mask)[0]
            octpath = octpath[filter_idx]
            octlevel = octlevel[filter_idx]
            unit_val = unit_val[filter_idx]

            # subdivide
            mask_sub = (octlevel.squeeze() < target_level)
            kept_idx = torch.where(~mask_sub)[0]
            subdiv_idx = torch.where(mask_sub)[0]
            if len(subdiv_idx) == 0:
                break

            child_octpath, child_octlevel = octree_utils.gen_children(octpath[subdiv_idx], octlevel[subdiv_idx])
            child_unit_val = subdivide_by_interp(unit_val[subdiv_idx])

            # merge + ?��계산
            octpath = torch.cat([octpath[kept_idx], child_octpath])
            octlevel = torch.cat([octlevel[kept_idx], child_octlevel])
            unit_val = torch.cat([unit_val[kept_idx], child_unit_val])
            grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath, octlevel)
            grid_pts_xyz = octree_utils.compute_gridpoints_xyz(grid_pts_key, voxel_model.scene_center, voxel_model.scene_extent)
            grid_sdf_vals = agg_voxel_into_grid_pts(len(grid_pts_xyz), vox_key, unit_val)

            print(f"[SDF] Subdiv {n_ori:10d} => {len(octlevel):10d}")

        del unit_val, octpath, octlevel

    else:
        unit_val = grid_sdf_vals[vox_key]
        grid_sdf_vals = agg_voxel_into_grid_pts(len(grid_pts_xyz), vox_key, unit_val)

    torch.cuda.empty_cache()

    # --------------------------
    # 6. Marching Cubes
    # --------------------------
    verts, faces = svraster_cuda.marching_cubes.torch_marching_cubes_grid(
        grid_pts_val=voxel_model._geo_grid_pts,
        grid_pts_xyz=grid_pts_xyz,
        vox_key=voxel_model.vox_key,
        iso=iso
    )
    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh


def direct_mc(args, voxel_model, final_lv, crop_bbox):
    # Filter background voxels
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")
    inside_mask = ((inside_min <= voxel_model.grid_pts_xyz) & (voxel_model.grid_pts_xyz <= inside_max)).all(-1)
    inside_mask = inside_mask[voxel_model.vox_key].any(-1)
    inside_idx = torch.where(inside_mask)[0]

    # Infer iso value for level set
    vox_level = torch.tensor([voxel_model.outside_level + final_lv], device="cuda")
    vox_size = octree_utils.level_2_vox_size(voxel_model.scene_extent, vox_level).item()
    iso_alpha = torch.tensor(0.5, device="cuda")
    iso_density = activation_utils.alpha2density(iso_alpha, vox_size)
    iso = getattr(activation_utils, f"{voxel_model.density_mode}_inverse")(iso_density)
    sign = -1

    verts, faces = svraster_cuda.marching_cubes.torch_marching_cubes_grid(
        grid_pts_val=sign * voxel_model._geo_grid_pts,
        grid_pts_xyz=voxel_model.grid_pts_xyz,
        vox_key=voxel_model.vox_key[inside_idx],
        iso=sign * iso)
    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh


def colorize_pts(args, pts, data_pack):
    cloest_color = torch.full([len(pts), 3], 0.5, dtype=torch.float32, device="cuda")
    cloest_dist = torch.full([len(pts)], np.inf, dtype=torch.float32, device="cuda")

    cam_lst = data_pack.get_train_cameras()

    for cam in tqdm(cam_lst):

        render_pkg = voxel_model.render(cam, color_mode="sh0", output_depth=True, output_T=True)
        frame_color = render_pkg['color']
        if args.use_mean:
            frame_depth = render_pkg['raw_depth'][[0]]  # Use mean depth
        else:
            frame_depth = render_pkg['raw_depth'][[2]]  # Use median depth
        frame_alpha = 1 - render_pkg['raw_T']
        H, W = frame_depth.shape[-2:]

        # Project grid points to image
        pts_uv = cam.project(pts)

        # Filter points projected outside
        filter_idx = torch.where((pts_uv.abs() <= 1).all(-1))[0]
        valid_pts_idx = filter_idx
        valid_pts = pts[filter_idx]
        pts_uv = pts_uv[filter_idx]

        # Sample alpha and filter
        pts_frame_alpha = torch.nn.functional.grid_sample(
            frame_alpha.view(1,1,H,W),
            pts_uv.view(1,1,-1,2),
            mode='bilinear',
            align_corners=False).flatten()
        filter_idx = torch.where(pts_frame_alpha > args.alpha_thres)[0]
        valid_pts_idx = valid_pts_idx[filter_idx]
        valid_pts = valid_pts[filter_idx]
        pts_uv = pts_uv[filter_idx]

        # Compute projective sdf
        pts_frame_depth = torch.nn.functional.grid_sample(
            frame_depth.view(1,1,H,W),
            pts_uv.view(1,1,-1,2),
            mode='bilinear',
            align_corners=False).flatten()
        pts_depth = ((valid_pts - cam.position) @ cam.lookat)
        pts_dist = (pts_frame_depth - pts_depth).abs()

        filter_idx = torch.where(pts_dist < cloest_dist[valid_pts_idx])[0]
        valid_pts_idx = valid_pts_idx[filter_idx]
        pts_uv = pts_uv[filter_idx]
        pts_dist = pts_dist[filter_idx]
        pts_color = torch.nn.functional.grid_sample(
            frame_color[None],
            pts_uv.view(1,1,-1,2),
            mode='bilinear',
            align_corners=False).squeeze().T
        cloest_dist[valid_pts_idx] = pts_dist
        cloest_color[valid_pts_idx] = pts_color

    return cloest_color


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster extract mesh.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_gpu", action='store_true')
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None, type=str)
    parser.add_argument("--bbox_path", default=None)
    parser.add_argument("--bbox_scale", default=1.0, type=float)
    parser.add_argument("--mesh_fname", default=None, type=str)

    parser.add_argument("--direct", action='store_true')
    parser.add_argument("--adaptive", action='store_true')
    parser.add_argument("--init_lv", default=7, type=int)
    parser.add_argument("--final_lv", default=9, type=int)
    parser.add_argument("--trunc_lv", default=10, type=int)
    parser.add_argument("--trunc_vox", default=5.0, type=float)
    parser.add_argument("--crop_border", default=0.01, type=float)
    parser.add_argument("--alpha_thres", default=0.5, type=float)
    parser.add_argument("--pg_prune", default=0.6, type=float)
    parser.add_argument("--use_mean", action='store_true')
    parser.add_argument("--use_vert_color", action='store_true')
    parser.add_argument("--use_clean", action='store_true')
    parser.add_argument("--use_lv_avg", action='store_true')
    parser.add_argument("--use_remesh", action='store_true')
    parser.add_argument("--remesh_len", default=-1, type=float)

    parser.add_argument("--voxel_size", default=0.004, type=float)
    parser.add_argument("--sdf_trunc", default=0.016, type=float)
    parser.add_argument("--depth_trunc", default=3.0, type=float)
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    # Load data
    data_pack = DataPack(cfg.data, cfg.model.white_background)

    # Load model
    voxel_model = SparseVoxelModel(cfg.model)
    voxel_model.load_iteration(args.iteration)
    voxel_model.freeze_vox_geo()

    if args.overwrite_ss is not None:
        voxel_model.ss = args.overwrite_ss
    if args.overwrite_vox_geo_mode is not None:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode

    # Prepare output dir
    outdir = os.path.join(
        args.model_path, "mesh",
        f"iter{voxel_model.loaded_iter:06d}" if voxel_model.loaded_iter > 0 else "latest")
    os.makedirs(outdir, exist_ok=True)

    print(f'outdir: {outdir}')
    print(f'ss            =: {voxel_model.ss}')
    print(f'vox_geo_mode  =: {voxel_model.vox_geo_mode}')
    print(f'density_mode  =: {voxel_model.density_mode}')

    # Read crop bbox
    if args.bbox_path:
        crop_bbox = np.loadtxt(args.bbox_path)
    else:
        crop_bbox = None

    # GOGO
    fname = 'mesh'
    eps_time = time.time()
    with torch.no_grad():
        if args.direct:
            mesh = direct_mc(args, voxel_model, args.final_lv, crop_bbox)
            fname += f'_direct'
        elif voxel_model.density_mode == 'sdf':
            mesh = extract_mesh_sdf(args, voxel_model, args.final_lv, crop_bbox, use_lv_avg=False, iso=0.0)
            fname += f'_lv{args.final_lv}_sdf_lv_avg'
        elif args.adaptive:
            mesh = extract_mesh(args, data_pack, voxel_model, args.final_lv, crop_bbox, args.use_lv_avg)
            fname += f'_lv{args.final_lv}_adaptive'
            if args.use_lv_avg:
                fname += '_lv_avg'
        else:
            fname += f'_lv{args.init_lv}-{args.final_lv}'
            mesh = extract_mesh_progressive(args, data_pack, voxel_model, args.init_lv, args.final_lv, crop_bbox)
    eps_time = time.time() - eps_time
    print(f"Extracted mesh in {eps_time:.3f} sec")

    if args.use_mean:
        fname += '_dmean'

    # Taking the biggest connected component
    if args.use_clean:
        fname += '_clean'
        print("Taking the biggest connected component")
        try:
            labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
            cc, cc_cnt = np.unique(labels, return_counts=True)
            cc_maxid = cc[cc_cnt.argmax()]
            mesh.update_faces(labels==cc_maxid)

            vmask = np.zeros([len(mesh.vertices)], dtype=bool)
            vmask[mesh.faces] = 1
            mesh.update_vertices(vmask)
        except:
            print("Failed to segment largest cc")

    # Remesh
    if args.use_remesh:
        from gpytoolbox import remesh_botsch
        avg_edge_len = mesh.edges_unique_length.mean()
        if args.remesh_len < 0:
            target_edge_len = min(avg_edge_len, voxel_model.inside_extent.item() / 1024)
        else:
            target_edge_len = args.remesh_len
        print(f"Remeshing: original avg_len={avg_edge_len}; target edge_len={target_edge_len}")
        try:
            eps_time = time.time()
            v, f = remesh_botsch(mesh.vertices, mesh.faces, i=5, h=target_edge_len)
            eps_time = time.time() - eps_time
            print(f"Remeshed in {eps_time:.3f} sec")
            mesh = trimesh.Trimesh(vertices=v, faces=f)
        except:
            print(f"Remesh failed.")

    # Colorize vertices
    # TODO: Unwrap and use high-res UV texture map
    verts_color = None
    if args.use_vert_color:
        print("Colorizing vertices")
        with torch.no_grad():
            pts = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
            verts_color = colorize_pts(args, pts, data_pack)
            verts_color = verts_color.cpu().numpy()
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=verts_color)

    # Transform to world coordinate
    if data_pack.to_world_matrix is not None:
        mesh = mesh.apply_transform(data_pack.to_world_matrix)

    # Export mesh
    print(mesh)
    if args.mesh_fname is not None:
        fname = args.mesh_fname
    outpath = os.path.join(outdir, f'{fname}.ply')
    mesh.export(outpath)
    print('Save to', outpath)

    print("done!")
