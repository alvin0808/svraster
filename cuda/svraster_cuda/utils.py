# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C


def voxel_order_rank(octree_paths):
    # Compute the eight possible voxel rendering orders.
    order_ranks = _C.voxel_order_rank(octree_paths)
    return order_ranks


def is_in_cone(pts, cam):
    assert torch.is_tensor(pts)
    assert pts.device == cam.w2c.device
    assert len(pts.shape) == 2
    assert pts.shape[1] == 3
    return _C.is_in_cone(
        cam.tanfovx,
        cam.tanfovy,
        cam.near,
        cam.w2c,
        pts)


def compute_rd(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix):
    assert torch.is_tensor(c2w_matrix)
    return _C.compute_rd(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix)


def depth2pts(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix, depth):
    assert torch.is_tensor(c2w_matrix)
    assert depth.device == c2w_matrix.device
    assert depth.numel() == width * height
    return _C.depth2pts(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix, depth)


def ijk_2_octpath(ijk, octlevel):
    assert torch.is_tensor(ijk) and torch.is_tensor(octlevel)
    assert len(ijk.shape) == 2 and ijk.shape[1] == 3
    assert ijk.numel() == octlevel.numel() * 3
    assert ijk.dtype == torch.int64
    assert octlevel.dtype == torch.int8
    return _C.ijk_2_octpath(ijk, octlevel)


def octpath_2_ijk(octpath, octlevel):
    assert torch.is_tensor(octpath) and torch.is_tensor(octlevel)
    assert octpath.numel() == octlevel.numel()
    assert octpath.dtype == torch.int64
    assert octlevel.dtype == torch.int8
    return _C.octpath_2_ijk(octpath, octlevel)

def valid_gradient_table(vox_center, vox_size, scene_center, inside_extent, grid_res, is_leaf):
    '''
    grid_res = 2**grid_res
    inside_min = scene_center - 0.5 * inside_extent
    inside_max = scene_center + 0.5 * inside_extent
    inside_mask = ((inside_min <= vox_center) & (vox_center <= inside_max)).all(-1)
    flat_size = grid_res **3
    grid_mask = torch.zeros((flat_size,), dtype=torch.bool, device='cuda')
    grid_keys = []
    grid2voxel = []
    for vox_idx in range(len(vox_center)):
        if not inside_mask[vox_idx]:
            continue
        
        #print(f"Processing voxel {vox_idx} at {vox_center.tolist()} with size {vox_size.tolist()}")
        min_corner = vox_center[vox_idx] - 0.5 * vox_size[vox_idx]
        max_corner = vox_center[vox_idx] + 0.5 * vox_size[vox_idx]
        min_grid_pos = (min_corner - inside_min) / (inside_max - inside_min) * grid_res
        max_grid_pos = (max_corner - inside_min) / (inside_max - inside_min) * grid_res
        x0, y0, z0 =  map(int,min_grid_pos.tolist())
        x1, y1, z1 =  map(int,max_grid_pos.tolist())
        #print(f"Voxel {vox_idx} at {self.octpath[vox_idx].tolist()} with level {self.octlevel[vox_idx].item()} has grid range: "
        #      f"({x0}, {y0}, {z0}) to ({x1}, {y1}, {z1})")
        # Ensure the range is within bounds
        for xi in range(x0, x1):
            for yi in range(y0, y1):
                for zi in range(z0, z1):
                    flat_idx = xi + grid_res * (yi + grid_res * zi)
                    grid_mask[flat_idx] = True
                    grid_keys.append(flat_idx)
                    grid2voxel.append(vox_idx)
    print(f"Total valid grid points: {len(grid_keys)}")
    print(f"Total valid voxels: {len(grid2voxel)}")
    # Convert list to tensor
    grid_keys = torch.tensor(grid_keys, dtype=torch.int32, device="cuda")
    grid2voxel = torch.tensor(grid2voxel, dtype=torch.int32, device="cuda")
    return grid_mask, grid_keys, grid2voxel
    '''
    return _C.valid_gradient_table(vox_center, vox_size, scene_center, inside_extent, grid_res, is_leaf)