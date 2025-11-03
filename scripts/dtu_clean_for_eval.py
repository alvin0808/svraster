# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import cv2
import glob
import trimesh
import numpy as np
from tqdm import trange


if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Clean mesh for evaluation.")
    parser.add_argument('data_dir')
    parser.add_argument('mesh_path')
    args = parser.parse_args()

    # Read mesh
    mesh = trimesh.load(args.mesh_path)
    print("Loaded mesh:", mesh)

    # Start cleaning
    print('Running DTU_clean_mesh_by_mask...')
    verts = np.copy(mesh.vertices[:])
    faces = np.copy(mesh.faces[:])
    cameras = np.load(f'{args.data_dir}/cameras_sphere.npz')
    mask_lis = sorted(glob.glob(f'{args.data_dir}/mask/*.png'))

    n_images = len(mask_lis)
    mask = np.ones(len(verts), dtype=bool)
    for i in trange(n_images):
        P = cameras[f'world_mat_{i}']
        pts_image = np.matmul(P[None, :3, :3], verts[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1
        mask_image = cv2.imread(mask_lis[i])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask_image = cv2.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)
        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)
        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]
        mask &= curr_mask.astype(bool)

    print('Valid vertices ratio:', mask.mean())

    indexes = np.full(len(verts), -1, dtype=np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
    new_faces = faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = verts[np.where(mask)]

    mesh = trimesh.Trimesh(new_vertices, new_faces)
    try:
        print('Kept only the largest CC')
        meshes = mesh.split(only_watertight=False)
        mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]
    except:
        print('Failed')
    outdir, outfname = os.path.split(args.mesh_path)
    outfname = outfname[:-4] + '_cleaned_for_eval.ply'
    mesh.export(os.path.join(outdir, outfname))
