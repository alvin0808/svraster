# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import svraster_cuda

from src.utils import octree_utils
from src.utils.fuser_utils import rgb_fusion
from src.utils.activation_utils import rgb2shzero

class SVProperties:

    @property
    def num_voxels(self):
        return len(self.octpath)

    @property
    def num_grid_pts(self):
        return len(self.grid_pts_key)

    @property
    def scene_min(self):
        return self.scene_center - 0.5 * self.scene_extent

    @property
    def scene_max(self):
        return self.scene_center + 0.5 * self.scene_extent

    @property
    def inside_min(self):
        return self.scene_center - 0.5 * self.inside_extent

    @property
    def inside_max(self):
        return self.scene_center + 0.5 * self.inside_extent

    @property
    def inside_mask(self):
        isin = ((self.inside_min < self.vox_center) & (self.vox_center < self.inside_max)).all(1)
        return isin

    @property
    def sh0(self):
        return self._sh0

    @property
    def shs(self):
        return self._shs

    @property
    def signature(self):
        # Signature to check if the voxel grid layout is updated
        return (
            self.num_voxels, self.num_grid_pts,
            id(self.octpath), id(self.octlevel),
            id(self.vox_key), id(self.grid_pts_key))

    @property
    def vox_size_inv(self):
        # Lazy computation of inverse voxel sizes
        signature = self.signature
        need_recompute = not hasattr(self, '_vox_size_inv') or \
                         self._vox_size_log_signature != signature
        if need_recompute:
            self._vox_size_inv = 1 / self.vox_size
            self._vox_size_log_signature = signature
        return self._vox_size_inv

    @property
    def grid_pts_xyz(self):
        # Lazy computation of grid points xyz
        signature = self.signature
        need_recompute = not hasattr(self, '_grid_pts_xyz') or \
                         self._grid_pts_xyz_signature != signature
        if need_recompute:
            self._grid_pts_xyz = octree_utils.compute_gridpoints_xyz(
                self.grid_pts_key, self.scene_center, self.scene_extent)
            self._grid_pts_xyz_signature = signature
        return self._grid_pts_xyz

    @torch.no_grad()
    def reset_sh_from_cameras(self, cameras):
        self._sh0.data.copy_(rgb2shzero(rgb_fusion(self, cameras)))
        self._shs.data.zero_()
