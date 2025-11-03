# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
import json
from argparse import ArgumentParser
import glob
import numpy as np
import cv2
from natsort import natsorted
import math
from tqdm import tqdm
from PIL import Image

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def load_K_Rt_from_P(filename, P=None):
    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


parser = ArgumentParser(description="Training script parameters")
parser.add_argument('dataset_root')
args = parser.parse_args()

for scene in os.listdir(args.dataset_root):
    scene_path = os.path.join(args.dataset_root, scene)
    if not os.path.isdir(scene_path) or 'scan' not in scene:
        continue
    
    camera_param = dict(np.load(os.path.join(scene_path, 'cameras_sphere.npz')))
    images_lis = sorted(glob.glob(os.path.join(scene_path, 'image/*.png')))

    train = dict(camera_angle_x=0, frames=[])
    test = dict(camera_angle_x=0, frames=[])
    for idx, image in enumerate(images_lis):
        image = os.path.basename(image)
        stem = os.path.splitext(image)[0]

        world_mat = camera_param['world_mat_%d' % idx]
        scale_mat = camera_param['scale_mat_%d' % idx]

        # scale and decompose
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsic_param, c2w = load_K_Rt_from_P(None, P)

        fx = float(intrinsic_param[0][0])
        fy = float(intrinsic_param[1][1])
        cx = float(intrinsic_param[0][2])
        cy = float(intrinsic_param[1][2])
        w, h = Image.open(os.path.join(scene_path, 'image', image)).size
        camera_angle_x = focal2fov(fx, w)
        camera_angle_y = focal2fov(fy, h)

        # To synthetic blender format
        c2w[:3, 1:3] *= -1

        frame = {
            "file_path": 'image/' + stem,
            "mask_path": f'mask/{int(stem):03d}.png',
            "normal_path": f'normal/{int(stem):03d}.png',
            "camera_angle_x": camera_angle_x,
            "camera_angle_y": camera_angle_y,
            "cx_p": cx / w,
            "cy_p": cy / h,
            "transform_matrix": c2w.tolist()
        }
        if idx % 8 == 9:
            test['frames'].append(frame)
        else:
            train['frames'].append(frame)

    out_train_path = os.path.join(scene_path, 'transforms_train.json')
    out_test_path = os.path.join(scene_path, 'transforms_test.json')
    with open(out_train_path, 'w') as f:
        json.dump(train, f, indent=4)

    with open(out_test_path, 'w') as f:
        json.dump(test, f, indent=4)

    # Write down scene bound
    out_bound_path = os.path.join(scene_path, 'nerf_normalization.json')
    with open(out_bound_path, 'w') as f:
        json.dump({"center": [0.,0.,0.], "radius": 1.0}, f, indent=4)

    np.savetxt(
        os.path.join(scene_path, 'to_world_matrix.txt'),
        camera_param['scale_mat_0'])
