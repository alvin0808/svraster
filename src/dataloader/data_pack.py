# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from src.dataloader.reader_colmap_dataset import read_colmap_dataset
from src.dataloader.reader_nerf_dataset import read_nerf_dataset

from src.cameras import Camera, MiniCam

AUTO_MAX_N_PIXELS = 1024 ** 2
AUTO_WARN = False

class DataPack:

    def __init__(self, cfg_data, white_background=False, dataset_downscales=[1.0], camera_params_only=False):

        self._cameras = dict(train={}, test={})

        sparse_path = os.path.join(cfg_data.source_path, "sparse")
        colmap_path = os.path.join(cfg_data.source_path, "colmap", "sparse")
        meta_path1 = os.path.join(cfg_data.source_path, "transforms_train.json")
        meta_path2 = os.path.join(cfg_data.source_path, "transforms.json")

        if os.path.exists(sparse_path) or os.path.exists(colmap_path):
            print("Read dataset in COLMAP format.")
            scene_info = read_colmap_dataset(
                path=cfg_data.source_path,
                images=cfg_data.images,
                depth_paths=cfg_data.depth_paths,
                normal_paths=cfg_data.normal_paths,
                test_every=cfg_data.test_every,
                eval=cfg_data.eval)
        elif os.path.exists(meta_path1) or os.path.exists(meta_path2):
            print("Read dataset in NeRF format.")
            scene_info = read_nerf_dataset(
                path=cfg_data.source_path,
                extension=cfg_data.extension,
                test_every=cfg_data.test_every,
                eval=cfg_data.eval)
        else:
            raise Exception("Unknown scene type!")

        print(f"res_downscale={cfg_data.res_downscale}")
        print(f"res_width={cfg_data.res_width}")
        if camera_params_only:
            self._cameras['train'][1.0] = CameraList(
                scene_info.train_cam_infos, cfg_data, camera_params_only=True)
            self._cameras['test'][1.0] = CameraList(
                scene_info.test_cam_infos, cfg_data, camera_params_only=True)
        else:
            for dataset_downscale in dataset_downscales:
                self._cameras['train'][dataset_downscale] = CameraList(
                    scene_info.train_cam_infos, cfg_data, dataset_downscale)
                self._cameras['test'][dataset_downscale] = CameraList(
                    scene_info.test_cam_infos, cfg_data, dataset_downscale)

        self.has_depth = False
        self.has_mask = False
        self.has_normal = False
        self.has_conf = False
        for cams_split in self._cameras.values():
            for cams in cams_split.values():
                for cam in cams:
                    self.has_depth |= cam.depth is not None
                    self.has_mask |= cam.mask is not None
                    self.has_normal |= cam.normal is not None
                    self.has_conf |= cam.conf is not None

        if self.has_mask and cfg_data.blend_mask:
            bg_color = torch.tensor([float(white_background)]*3, dtype=torch.float32)
            self.composite_bg_color(bg_color)

        self.suggested_bounding = scene_info.suggested_bounding

        self.to_world_matrix = None
        to_world_path = os.path.join(cfg_data.source_path, 'to_world_matrix.txt')
        if os.path.isfile(to_world_path):
            self.to_world_matrix = np.loadtxt(to_world_path)

        self.point_cloud = scene_info.point_cloud
        self.sfm_init_data = scene_info.sfm_init_data
    def get_train_cameras(self, scale=1.0):
        return self._cameras['train'][scale]

    def get_test_cameras(self, scale=1.0):
        return self._cameras['test'][scale]

    def composite_bg_color(self, bg_color):
        for cams_split in self._cameras.values():
            for cams in cams_split.values():
                for cam in cams:
                    cam.composite_bg_color(bg_color)


def compute_iter_idx(num_data, num_iter):
    tr_iter_idx = []
    while len(tr_iter_idx) < num_iter:
        lst = list(range(num_data))
        random.shuffle(lst)
        tr_iter_idx.extend(lst)
    return tr_iter_idx[:num_iter]

def compute_iter_idx_sparse(num_data, num_iter, skip_num=1):
    tr_iter_idx = []
    while len(tr_iter_idx) < num_iter:
        lst = list(range(0, num_data, skip_num))
        random.shuffle(lst)
        tr_iter_idx.extend(lst)
    return tr_iter_idx[:num_iter]


class CameraList:
    def __init__(self, cam_infos, cfg_data, dataset_downscale=1.0, camera_params_only=False):
        if camera_params_only:
            self.camera_list = [
                instantiate_a_minicamera(cam_info, cfg_data, dataset_downscale)
                for cam_info in cam_infos
            ]
        else:
            self.camera_list = [
                instantiate_a_camera(cam_info, cfg_data, dataset_downscale)
                for cam_info in cam_infos
            ]
            for i in range(len(cam_infos)):
                self.camera_list[i] = self.camera_list[i].to(cfg_data.data_device)

    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        return self.camera_list[idx]


def instantiate_a_camera(cam_info, cfg_data, dataset_downscale):
    # Determine target resolution
    W, H = cam_info.image.size
    if cfg_data.res_downscale > 0:
        global_downscale = cfg_data.res_downscale
    elif cfg_data.res_width > 0:
        global_downscale = W / cfg_data.res_width
    elif W * H > AUTO_MAX_N_PIXELS and cfg_data.images == "images":
        global_downscale = (AUTO_MAX_N_PIXELS / (W * H)) ** -0.5
        global AUTO_WARN
        if not AUTO_WARN:
            AUTO_WARN = True
            print(f"[WARN] Source images are too large ({W}x{H}). ")
            print(f"       Auto downscale gt by {global_downscale}. ")
            print(f"       Use `--images`, `--res_downscale`, or `--res_width` to prevent it.")
    else:
        global_downscale = 1

    target_downscale = float(global_downscale * dataset_downscale)
    target_resolution = (round(W / target_downscale), round(H / target_downscale))

    # Resize image if needed
    if (W, H) != target_resolution:
        pil_image = cam_info.image.resize(target_resolution)
    else:
        pil_image = cam_info.image

    # Read color image
    gt_image = torch.tensor(np.array(pil_image), dtype=torch.float32).moveaxis(-1, 0) / 255.0
    mask = None
    if gt_image.shape[0] == 4:
        gt_image, mask = gt_image.split([3, 1], dim=0)

    # Load mask if exist
    if cam_info.mask is not None:
        if mask is not None:
            raise NotImplementedError("Duplicated mask from RGBA and given mask path !?")
        if cam_info.mask.size != target_resolution:
            pil_mask = cam_info.mask.resize(target_resolution)
        else:
            pil_mask = cam_info.mask
        mask = torch.tensor(np.array(pil_mask), dtype=torch.float32) / 255.0
        if len(mask.shape) == 3:
            mask = mask.mean(-1)
        mask = mask.unsqueeze(0).contiguous()

    # Load depth if exist
    depth = None
    if cam_info.depth is not None:
        if cam_info.depth.size != target_resolution:
            pil_depth = cam_info.depth.resize(target_resolution, Image.Resampling.NEAREST)
        else:
            pil_depth = cam_info.depth
        depth = torch.tensor(np.array(pil_depth) / cfg_data.depth_scale, dtype=torch.float32)
        depth = depth.unsqueeze(0).contiguous()
    normal = None
    if cam_info.normal is not None:
        if cam_info.normal.size != target_resolution:
            pil_normal = cam_info.normal.resize(target_resolution, Image.Resampling.BILINEAR)
        else:
            pil_normal = cam_info.normal

        n = np.asarray(pil_normal, dtype=np.float32) / 255.0     # [H,W,3] in [0,1]
        n = n * 2.0 - 1.0                                        # → [-1,1]
        n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)

        R_wc = cam_info.w2c[:3, :3].T.astype(np.float32)         # [3,3]
        R_wc_t = torch.from_numpy(R_wc)                           # torch [3,3]

        normal_cam = torch.from_numpy(n).permute(2,0,1).contiguous().float()  # [3,H,W]
        normal_world = torch.einsum('ij,jhw->ihw', R_wc_t, normal_cam)        # [3,H,W]

        normal = F.normalize(normal_world, dim=0, eps=1e-8)

    return Camera(w2c=cam_info.w2c,
                  fovx=cam_info.fovx, fovy=cam_info.fovy,
                  cx_p=cam_info.cx_p, cy_p=cam_info.cy_p,
                  image=gt_image, mask=mask, depth=depth, normal=normal,
                  sparse_pt=cam_info.sparse_pt,
                  image_name=cam_info.image_name)


def instantiate_a_minicamera(cam_info, cfg_data, dataset_downscale=1):
    # Determine target resolution
    W, H = cam_info.image.size
    if cfg_data.res_downscale > 0:
        global_downscale = cfg_data.res_downscale
    elif cfg_data.res_width > 0:
        global_downscale = W / cfg_data.res_width
    elif W * H > AUTO_MAX_N_PIXELS and cfg_data.images == "images":
        global_downscale = (AUTO_MAX_N_PIXELS / (W * H)) ** -0.5
        global AUTO_WARN
        if not AUTO_WARN:
            AUTO_WARN = True
            print(f"[WARN] Auto downscale gt by {global_downscale}. ")
            print(f"       Use `--res_downscale` or `--res_width` to prevent it.")
    else:
        global_downscale = 1

    target_downscale = float(global_downscale * dataset_downscale)
    target_resolution = (round(W / target_downscale), round(H / target_downscale))

    return MiniCam(
        c2w=np.linalg.inv(cam_info.w2c),
        fovx=cam_info.fovx,
        fovy=cam_info.fovy,
        width=target_resolution[0],
        height=target_resolution[1],
        cx_p=cam_info.cx_p, cy_p=cam_info.cy_p)
