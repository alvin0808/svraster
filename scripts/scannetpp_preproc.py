# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import math
import json
import argparse
import natsort
import collections
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm


Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1
    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    points_id = np.empty((num_points), dtype=np.int32)
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                points_id[count] = elems[0]
                count += 1
    return xyzs, rgbs, errors, points_id


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='data/scanetpp/data')
    parser.add_argument('--outdir', default='data/scannetpp_nvs')
    parser.add_argument('--ids', default=['39f36da05b', '5a269ba6fe', 'dc263dfbf0', '08bbbdcc3d'], nargs='+')
    #parser.add_argument('--ids', default=['7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', 'fb5a96b1a2', 'a24f64f7fb', '1ada7a0617', '5eb31827b7', '3e8bba0176', '3f15a9266d', '21d970d8de', '5748ce6f01', 'c4c04e6d6c', '7831862f02', 'bde1e479ad', '38d58a7a31', '5ee7c22ba0', 'f9f95681fd', '3864514494', '40aec5fffa', '13c3e046d7', 'e398684d27', 'a8bf42d646', '45b0dac5e3', '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6', 'f3685d06a9', 'b0a08200c9', '825d228aec', 'a980334473', 'f2dc06b1d2', '5942004064', '25f3b7a318', 'bcd2436daf', 'f3d64c30f8', '0d2ee665be', '3db0a1c8f3', 'ac48a9b736', 'c5439f4607', '578511c8a9', 'd755b3d9d8', '99fa5c25e1', '09c1414f1b', '5f99900f09', '9071e139d9', '6115eddb86', '27dd4da69e', 'c49a8c6cff'], nargs='+')
    #parser.add_argument('--ids', default=['ca0e09014e', 'beb802368c', 'ebff4de90b', 'd228e2d9dd', '9e019d8be1', '11b696efba', '471cc4ba84', 'f20e7b5640', 'dfe9cbd72a', 'ccdc33dc2a', '124974734e', 'c0cbb1fea1', '047fb766c4', '7b37cccb03', '8283161f1b', 'c3e279be54', '5a14f9da39', 'cd7973d92b', '5298ec174f', 'e0e83b4ca3', '64ea6b73c2', 'f00bd5fa8a', '02a980c994', 'be91f7884d', '1c876c250f', '15155a88fb', '633f9a9f06', 'd6419f6478', 'f0b0a42ba3', 'a46b21d949', '74ff105c0d', '77596f5d2a', 'ecb5d01065', 'c9bf4c8b62', 'b074ca565a', '49c758655e', 'd4d2019f5d', '319787e6ec', '84b48f2614', 'bee11d6a41', '9a9e32c768', '9b365a9b68', '54e7ffaea3', '7d72f01865', '252652d5ba', '651dc6b4f1', '03f7a0e617', 'fe94fc30cf', 'd1b9dff904', '4bc04e0cde'], nargs='+')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for scene_id in tqdm(args.ids):
        in_scene_dir = os.path.join(args.indir, scene_id, 'dslr')
        out_scene_dir = os.path.join(args.outdir, scene_id)

        os.system(f"mkdir -p {out_scene_dir}")

        os.system(f"cp -r {os.path.join(in_scene_dir, 'undistorted_images')} {os.path.join(out_scene_dir, 'images')}")

        with open(os.path.join(in_scene_dir, 'nerfstudio', 'transforms_undistorted.json')) as f:
            meta = json.load(f)

        cx_p = meta['cx'] / meta['w']
        cy_p = meta['cy'] / meta['h']
        camera_angle_x = focal2fov(meta['fl_x'], meta['w'])
        camera_angle_y = focal2fov(meta['fl_y'], meta['h'])

        new_metas_lst = []
        for key in ['frames', 'test_frames']:
            heldout = (key == 'test_frames') and ('test' in args.outdir)
            new_metas_lst.append(dict(camera_angle_x=0, frames=[]))
            for frame in meta[key]:
                new_metas_lst[-1]['frames'].append({
                    'camera_angle_x': camera_angle_x,
                    'camera_angle_y': camera_angle_y,
                    'cx_p': cx_p,
                    'cy_p': cy_p,
                    'file_path': f"images/{frame['file_path']}",
                    'transform_matrix': frame['transform_matrix'],
                    'is_bad': frame['is_bad'],
                    'heldout': heldout,
                    'w': meta['w'],
                    'h': meta['h'],
                })

        new_train_meta, new_test_meta = new_metas_lst

        with open(os.path.join(out_scene_dir, 'transforms_train.json'), 'w') as f:
            json.dump(new_train_meta, f)
        with open(os.path.join(out_scene_dir, 'transforms_test.json'), 'w') as f:
            json.dump(new_test_meta, f)

        # Process COLMAP point cloud
        xyz, rgb, err, points_id = read_points3D_text(f"{in_scene_dir}/colmap/points3D.txt")
        xyz = np.copy(xyz)  # Seems the raw-to-undistorted also transform the entire coordinates
        xyz[:, [0,1]] = xyz[:, [1,0]]
        xyz[:, 2] *= -1
        storePly(
            os.path.join(out_scene_dir, "points3D.ply"),
            xyz, rgb)

        # Process 2D-3D mapping
        cameras_extrinsic_file = os.path.join(in_scene_dir, "colmap", "images.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        points_idmap = np.full([points_id.max()+2], -1, dtype=np.int32)
        points_idmap[points_id] = np.arange(len(xyz))

        keys = natsort.natsorted(
            cam_extrinsics.keys(),
            key = lambda i : cam_extrinsics[i].name)

        correspondent = {}
        for idx, key in enumerate(keys):
            extr = cam_extrinsics[key]
            pt_idx = extr.point3D_ids
            pt_mask = (pt_idx != -1) & (points_idmap[pt_idx] != -1)
            correspondent[extr.name] = points_idmap[pt_idx[pt_mask]].tolist()
        with open(os.path.join(out_scene_dir, 'points_correspondent.json'), 'w') as f:
            json.dump(correspondent, f)
