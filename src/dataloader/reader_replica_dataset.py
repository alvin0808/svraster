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
import json
import yaml
import numpy as np
from PIL import Image
from pathlib import Path

from src.utils.camera_utils import fov2focal, focal2fov
from .colmap_loader import fetchPly
from .reader_scene_info import CameraInfo, PointCloud, SceneInfo


def read_replica_dataset(cfg_yaml_path, extension):
    """
    Reads a Replica-style dataset using a YAML config path (e.g., cfg/room0.yaml).
    That config contains:
      - inherit_from: path to replica.yaml (intrinsics, depth scale)
      - data.input_folder: where result/*.png and traj.txt live
      - mapping.bound: bounding box
    """

    # === 1. Load cfg/room0.yaml ===
    with open(cfg_yaml_path) as f:
        cfg = yaml.safe_load(f)

    # === 2. Load inherited config (e.g., cfg/replica.yaml) ===
    inherit_path = cfg["inherit_from"]
    with open(inherit_path) as f:
        inherit_cfg = yaml.safe_load(f)

    cam_cfg = inherit_cfg["cam"]
    W, H = cam_cfg["W"], cam_cfg["H"]
    fx, fy = cam_cfg["fx"], cam_cfg["fy"]
    cx, cy = cam_cfg["cx"], cam_cfg["cy"]
    png_depth_scale = cam_cfg["png_depth_scale"]

    # === 3. Get data path ===
    input_folder = os.path.normpath(cfg["data"]["input_folder"])
    result_path = os.path.join(input_folder, "results")
    traj_path = os.path.join(input_folder, "traj.txt")

    # === 4. Load poses ===
    poses = np.loadtxt(traj_path, dtype=np.float32).reshape(-1, 4, 4)

    # === 5. Build CameraInfo list ===
    cam_infos = []
    num_frames = len(poses)

    for idx in range(num_frames):
        rgb_path = os.path.join(result_path, f"frame{idx:06d}.jpg")
        depth_path = os.path.join(result_path, f"depth{idx:06d}.png")

        # Load images
        image = Image.open(rgb_path)
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img).astype(np.float32) / png_depth_scale
        depth = Image.fromarray(depth)

        # Convert OpenGL c2w → COLMAP-style w2c
        c2w = poses[idx]
        #c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w).astype(np.float32)

        cam_info = CameraInfo(
            image_name=f"frame{idx:05d}",
            w2c=w2c,
            fovx=2 * np.arctan(W / (2 * fx)),
            fovy=2 * np.arctan(H / (2 * fy)),
            width=W,
            height=H,
            cx_p=cx / W,
            cy_p=cy / H,
            image=image,
            image_path=rgb_path,
            depth=depth,
            depth_path=depth_path,
            mask=None,
            mask_path="",
            sparse_pt=None,
        )
        cam_infos.append(cam_info)

    # === 6. Bounding box ===
    mapping_cfg = cfg.get("mapping", {})
    raw_bounds = np.array(mapping_cfg["bound"], dtype=np.float32)  # shape (3, 2)

    min_xyz = raw_bounds[:, 0]
    max_xyz = raw_bounds[:, 1]

    center = (min_xyz + max_xyz) / 2
    half_lengths = (max_xyz - min_xyz) / 2
    max_half = np.max(half_lengths)

    # Create cube around center with edge length = 2 * max_half
    suggested_bounding = np.stack([
        center - max_half,
        center + max_half
    ]).astype(np.float32)  # shape (2, 3)

    # === 7. Pack scene info ===
    scene_info = SceneInfo(
        train_cam_infos=cam_infos,
        test_cam_infos=[],
        suggested_bounding=suggested_bounding,
        point_cloud=None
    )

    return scene_info