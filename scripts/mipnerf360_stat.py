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
import pandas as pd

parser = ArgumentParser(description="Training script parameters")
parser.add_argument('result_root')
parser.add_argument('--suffix', default='')
args = parser.parse_args()

indoor_scenes = ['bonsai', 'counter', 'kitchen', 'room']
outdoor_scenes = ['bicycle', 'garden', 'stump', 'treehill', 'flowers']
scenes = indoor_scenes + outdoor_scenes

indoor_psnr = []
indoor_ssim = []
indoor_lpips = []
indoor_lpips_corr = []
indoor_tr_time = []
indoor_fps = []
indoor_n_voxels = []
for scene in indoor_scenes:
    path = glob.glob(os.path.join(f'{args.result_root}/{scene}/test_stat/iter*.json'))
    if len(path) == 0:
        print(f'{scene:10s}: failed !!??')
        continue
    path = sorted(path)[-1]
    with open(path) as f:
        ret = json.load(f)
        tmp_psnr = ret['psnr']
        indoor_tr_time.append(ret['elapsed'] / 1000)
        # indoor_fps.append(ret['fps'])
        indoor_n_voxels.append(ret['n_voxels'])
    n_iter = int(os.path.split(path)[1].replace('iter', '').replace('.json', ''))
    fps_path = f'{args.result_root}/{scene}/test/ours_{n_iter}{args.suffix}.txt'
    with open(fps_path) as f:
        fps = float(f.read().strip().split()[-1].split('=')[1])
        indoor_fps.append(fps)
    eval_path = f'{args.result_root}/{scene}/results.json'
    if os.path.exists(eval_path):
        with open(os.path.join(eval_path)) as f:
            ret = json.load(f)
            ret = ret[f"ours_{n_iter}{args.suffix}"]
            indoor_psnr.append(ret['PSNR'])
            indoor_ssim.append(ret['SSIM'])
            indoor_lpips.append(ret['LPIPS'])
            indoor_lpips_corr.append(ret.get('LPIPS-corrected', 1))
    else:
        indoor_psnr.append(tmp_psnr)
        indoor_ssim.append(0)
        indoor_lpips.append(0)
        indoor_lpips_corr.append(1)

outdoor_psnr = []
outdoor_ssim = []
outdoor_lpips = []
outdoor_lpips_corr = []
outdoor_tr_time = []
outdoor_fps = []
outdoor_n_voxels = []
for scene in outdoor_scenes:
    path = glob.glob(os.path.join(f'{args.result_root}/{scene}/test_stat/iter*.json'))
    if len(path) == 0:
        print(f'{scene:10s}: failed !!??')
        continue
    path = sorted(path)[-1]
    with open(path) as f:
        ret = json.load(f)
        tmp_psnr = ret['psnr']
        outdoor_tr_time.append(ret['elapsed'] / 1000)
        # outdoor_fps.append(ret['fps'])
        outdoor_n_voxels.append(ret['n_voxels'])
    n_iter = int(os.path.split(path)[1].replace('iter', '').replace('.json', ''))
    fps_path = f'{args.result_root}/{scene}/test/ours_{n_iter}{args.suffix}.txt'
    with open(fps_path) as f:
        fps = float(f.read().strip().split()[-1].split('=')[1])
        outdoor_fps.append(fps)
    eval_path = f'{args.result_root}/{scene}/results.json'
    if os.path.exists(eval_path):
        with open(os.path.join(eval_path)) as f:
            ret = json.load(f)
            ret = ret[f"ours_{n_iter}{args.suffix}"]
            outdoor_psnr.append(ret['PSNR'])
            outdoor_ssim.append(ret['SSIM'])
            outdoor_lpips.append(ret['LPIPS'])
            outdoor_lpips_corr.append(ret.get('LPIPS-corrected', 1))
    else:
        outdoor_psnr.append(tmp_psnr)
        outdoor_ssim.append(0)
        outdoor_lpips.append(0)
        outdoor_lpips_corr.append(1)



def format_df_string(df):
    df = df.copy()
    df['scene'] = df['scene'].map(lambda s: s.rjust(15))
    df['psnr'] = df['psnr'].round(2)
    df['ssim'] = df['ssim'].round(3)
    df['lpips'] = df['lpips'].round(3)
    df['lpips*'] = df['lpips*'].round(3)
    df['tr-mins'] = (df['tr-mins'] / 60).round(1)
    df['fps'] = df['fps'].round(1)
    df['#vox(M)'] = (df['#vox(M)'] / 1_000_000).round(1)
    return df.to_string(index=False)

def add_avg_row(df):
    df_avg = df.mean(axis=0, numeric_only=True).to_frame().transpose()
    df_avg['scene'] = 'AVG'
    return pd.concat([df, df_avg], ignore_index=True)

df_indoor = pd.DataFrame({
    'scene': indoor_scenes,
    'psnr': indoor_psnr,
    'ssim': indoor_ssim,
    'lpips': indoor_lpips,
    'lpips*': indoor_lpips_corr,
    'tr-mins': indoor_tr_time,
    'fps': indoor_fps,
    '#vox(M)': indoor_n_voxels,
})

df_outdoor = pd.DataFrame({
    'scene': outdoor_scenes,
    'psnr': outdoor_psnr,
    'ssim': outdoor_ssim,
    'lpips': outdoor_lpips,
    'lpips*': outdoor_lpips_corr,
    'tr-mins': outdoor_tr_time,
    'fps': outdoor_fps,
    '#vox(M)': outdoor_n_voxels,
})

df = pd.concat([df_indoor, df_outdoor], ignore_index=True)

df_indoor = add_avg_row(df_indoor)
df_outdoor = add_avg_row(df_outdoor)
df = add_avg_row(df)

print(format_df_string(df_indoor))
print()
print(format_df_string(df_outdoor))
print()
print(format_df_string(df))

