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

import numpy as np

import torch
import svraster_cuda


class CameraBase:

    '''
    Base class of perspective cameras.
    '''

    def __repr__(self):
        clsname = self.__class__.__name__
        fname = f"image_name='{self.image_name}'"
        res = f"HW=({self.image_height}x{self.image_width})"
        fov = f"fovx={np.rad2deg(self.fovx):.1f}deg"
        return f"{clsname}({fname}, {res}, {fov})"

    @property
    def lookat(self):
        return self.c2w[:3, 2]

    @property
    def position(self):
        return self.c2w[:3, 3]

    @property
    def down(self):
        return self.c2w[:3, 1]

    @property
    def right(self):
        return self.c2w[:3, 0]

    @property
    def cx(self):
        return self.image_width * self.cx_p

    @property
    def cy(self):
        return self.image_height * self.cy_p

    @property
    def pix_size(self):
        return 2 * self.tanfovx / self.image_width

    @property
    def tanfovx(self):
        return np.tan(self.fovx * 0.5)

    @property
    def tanfovy(self):
        return np.tan(self.fovy * 0.5)

    def compute_rd(self, wh=None, cxcy=None, device=None):
        '''Ray directions in world space.'''
        if wh is None:
            wh = (self.image_width, self.image_height)
        if cxcy is None:
            cxcy = (self.cx * wh[0] / self.image_width, self.cy * wh[1] / self.image_height)
        rd = svraster_cuda.utils.compute_rd(
            width=wh[0], height=wh[1],
            cx=cxcy[0], cy=cxcy[1],
            tanfovx=self.tanfovx, tanfovy=self.tanfovy,
            c2w_matrix=self.c2w.cuda())
        rd = rd.to(device if device is None else self.c2w.device)
        return rd

    def project(self, pts, return_depth=False):
        # Return normalized image coordinate in [-1, 1]
        cam_pts = pts @ self.w2c[:3, :3].T + self.w2c[:3, 3]
        depth = cam_pts[:, [2]]
        cam_uv = cam_pts[:, :2] / depth
        scale_x = 1 / self.tanfovx
        scale_y = 1 / self.tanfovy
        shift_x = 2 * self.cx_p - 1
        shift_y = 2 * self.cy_p - 1
        cam_uv[:, 0] = cam_uv[:, 0] * scale_x + shift_x
        cam_uv[:, 1] = cam_uv[:, 1] * scale_y + shift_y
        if return_depth:
            return cam_uv, depth
        return cam_uv

    def depth2pts(self, depth):
        device = depth.device
        h, w = depth.shape[-2:]
        rd = self.compute_rd(wh=(w, h), device=device)
        return self.position.view(3,1,1).to(device) + rd * depth

    def depth2normal(self, depth, ks=3, tol_cos=-1):
        assert ks % 2 == 1
        pad = ks // 2
        ks_1 = ks - 1
        pts = self.depth2pts(depth)
        normal_pseudo = torch.zeros_like(pts)
        dx = pts[:, pad:-pad, ks_1:] - pts[:, pad:-pad, :-ks_1]
        dy = pts[:, ks_1:, pad:-pad] - pts[:, :-ks_1, pad:-pad]
        normal_pseudo[:, pad:-pad, pad:-pad] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=0), dim=0)

        if tol_cos > 0:
            with torch.no_grad():
                pts_dir = torch.nn.functional.normalize(pts - self.position.view(3,1,1), dim=0)
                dot = (normal_pseudo * pts_dir).sum(0)
                mask = (dot > tol_cos)
            normal_pseudo = normal_pseudo * mask

        return normal_pseudo

    def composite_bg_color(self, bg_color):
        if self.mask is None:
            return
        bg_color = bg_color.view(3, 1, 1).to(self.image.device)
        self.image = self.image * self.mask + (1 - self.mask) * bg_color


class Camera(CameraBase):
    def __init__(
            self, image_name,
            w2c, fovx, fovy, cx_p, cy_p,
            near=0.02,
            image=None, mask=None, depth=None, normal=None, conf=None,
            sparse_pt=None):

        self.image_name = image_name
        self.cam_mode = 'persp'

        # Camera parameters
        self.w2c = torch.tensor(w2c, dtype=torch.float32, device="cuda")
        self.c2w = self.w2c.inverse().contiguous()

        self.fovx = fovx
        self.fovy = fovy

        # Load frame
        self.image = image.cpu()

        # Other camera parameters
        self.image_width = self.image.shape[2]
        self.image_height = self.image.shape[1]
        self.cx_p = (0.5 if cx_p is None else cx_p)
        self.cy_p = (0.5 if cy_p is None else cy_p)
        self.near = near

        # Load mask and depth if there are
        self.mask = mask.cpu() if mask is not None else None
        self.depth = depth.cpu() if depth is not None else None
        self.normal = normal.cpu() if normal is not None else None
        self.conf = conf.cpu() if conf is not None else None
        # Load sparse depth
        if sparse_pt is not None:
            self.sparse_pt = torch.tensor(sparse_pt, dtype=torch.float32, device="cpu")
        else:
            self.sparse_pt = None

    def to(self, device):
        self.image = self.image.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        if self.normal is not None:
            self.normal = self.normal.to(device)
        if self.conf is not None:
            self.conf = self.conf.to(device)
        return self

    def auto_exposure_init(self):
        self._exposure_A = torch.eye(3, dtype=torch.float32, device="cuda")
        self._exposure_t = torch.zeros([3,1,1], dtype=torch.float32, device="cuda")
        self.exposure_updated = False

    def auto_exposure_apply(self, image):
        if self.exposure_updated:
            image = torch.einsum('ij,jhw->ihw', self._exposure_A, image) + self._exposure_t
        return image

    def auto_exposure_update(self, ren, ref):
        self.exposure_updated = True
        self._exposure_A.requires_grad_()
        self._exposure_t.requires_grad_()
        optim = torch.optim.Adam([self._exposure_A, self._exposure_t], lr=1e-3)
        for _ in range(100):
            loss = (self.auto_exposure_apply(ren).clamp(0, 1) - ref).abs().mean()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        self._exposure_A.requires_grad_(False)
        self._exposure_t.requires_grad_(False)

    def clone_mini(self):
        return MiniCam(
            c2w=self.c2w.clone(),
            fovx=self.fovx, fovy=self.fovy,
            width=self.image_width, height=self.image_height,
            near=self.near,
            cx_p=self.cx_p, cy_p=self.cy_p)

class MiniCam(CameraBase):
    def __init__(self,
            c2w, fovx, fovy,
            width, height,
            near=0.02,
            cx_p=None, cy_p=None):

        self.image_name = 'minicam'
        self.cam_mode = 'persp'
        self.c2w = torch.tensor(c2w).clone().cuda()
        self.w2c = self.c2w.inverse()

        self.fovx = fovx
        self.fovy = fovy
        self.image_width = width
        self.image_height = height
        self.cx_p = (0.5 if cx_p is None else cx_p)
        self.cy_p = (0.5 if cy_p is None else cy_p)
        self.near = near

        self.depth = None
        self.mask = None

    def clone_mini(self):
        return MiniCam(
            c2w=self.c2w.clone(),
            fovx=self.fovx, fovy=self.fovy,
            width=self.image_width, height=self.image_height,
            near=self.near,
            cx_p=self.cx_p, cy_p=self.cy_p)

    def move_forward(self, dist):
        new_position = self.position + dist * self.lookat
        self.c2w[:3, 3] = new_position
        self.w2c = self.c2w.inverse()
        return self

    def move_up(self, dist):
        return self.move_down(-dist)

    def move_down(self, dist):
        new_position = self.position + dist * self.down
        self.c2w[:3, 3] = new_position
        self.w2c = self.c2w.inverse()
        return self

    def move_right(self, dist):
        new_position = self.position + dist * self.right
        self.c2w[:3, 3] = new_position
        self.w2c = self.c2w.inverse()
        return self

    def move_left(self, dist):
        return self.move_right(-dist)

    def rotate(self, R):
        self.c2w[:3, :3] = (R @ self.w2c[:3, :3]).T
        self.w2c = self.c2w.inverse()
        return self

    def rotate_x(self, rad=None, deg=None):
        assert rad is None or deg is None, "Can only specify rotation by either rad or deg."
        if rad is None:
            rad = np.deg2rad(deg)
        R = torch.tensor([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)],
        ], dtype=torch.float32, device="cuda")
        return self.rotate(R)

    def rotate_y(self, rad=None, deg=None):
        assert rad is None or deg is None, "Can only specify rotation by either rad or deg."
        if rad is None:
            rad = np.deg2rad(deg)
        R = torch.tensor([
            [np.cos(rad), 0, -np.sin(rad)],
            [0, 1, 0],
            [np.sin(rad), 0, np.cos(rad)],
        ], dtype=torch.float32, device="cuda")
        return self.rotate(R)

    def rotate_z(self, rad=None, deg=None):
        assert rad is None or deg is None, "Can only specify rotation by either rad or deg."
        if rad is None:
            rad = np.deg2rad(deg)
        R = torch.tensor([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1],
        ], dtype=torch.float32, device="cuda")
        return self.rotate(R)


class OrthoCam(MiniCam):
    def __init__(self,
            c2w,
            x_len, y_len,
            width, height,
            near=0.02):

        self.image_name = 'orthocam'
        self.cam_mode = 'ortho'

        self.c2w = torch.tensor(c2w).clone()
        self.w2c = self.c2w.inverse()

        self.c2w = self.c2w.cuda()
        self.w2c = self.w2c.cuda()

        self.image_width = width
        self.image_height = height

        self.tanfovx = x_len * 0.5
        self.tanfovy = y_len * 0.5
        self.cx_p = 0.5
        self.cy_p = 0.5
        self.near = near

    def __repr__(self):
        clsname = self.__class__.__name__
        fname = f"image_name='{self.image_name}'"
        res = f"HW=({self.image_height}x{self.image_width})"
        fov = f"len={self.tanfovy*2:.1f}x{self.tanfovx*2}"
        return f"{clsname}({fname}, {res}, {fov})"
