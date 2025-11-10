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

import PIL
import numpy as np
from typing import Dict, List, NamedTuple

class CameraInfo(NamedTuple):
    image_name: str

    # Camera parameters
    w2c: np.array
    fovx: np.array
    fovy: np.array
    width: int
    height: int

    # (cx, cy) = (cx_p * width, cy_p * height)
    cx_p: float = None
    cy_p: float = None

    # Frame info
    image: PIL.Image.Image = None
    image_path: str = ""
    depth: PIL.Image.Image = None
    depth_path: str = ""
    normal: PIL.Image.Image = None
    normal_path: str = ""
    conf: PIL.Image.Image = None
    conf_path: str = ""
    mask: PIL.Image.Image = None
    mask_path: str = ""

    sparse_pt: np.array = None


class PointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    ply_path: str

class SFMInitData(NamedTuple):
    points_xyz: np.array
    index_to_point_id: np.array
    point_id_to_image_ids: Dict[int, np.array]

class SceneInfo(NamedTuple):
    train_cam_infos: List[CameraInfo]
    test_cam_infos: List[CameraInfo]
    suggested_bounding: np.array
    point_cloud: PointCloud
    sfm_init_data: SFMInitData = None
