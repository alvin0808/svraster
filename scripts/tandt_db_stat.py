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
args = parser.parse_args()

tandt_scenes = ['train', 'truck']
db_scenes = ['drjohnson', 'playroom']

tandt_psnr = []
tandt_ssim = []
tandt_lpips = []
tandt_tr_time = []
tandt_fps = []
tandt_n_voxels = []
for scene in tandt_scenes:
    path = glob.glob(os.path.join(f'{args.result_root}/{scene}/test_stat/iter*.json'))
    if len(path) == 0:
        print(f'{scene:10s}: failed !!??')
        continue
    path = sorted(path)[-1]
    with open(path) as f:
        ret = json.load(f)
        tmp_psnr = ret['psnr']
        tandt_tr_time.append(ret['elapsed'] / 1000)
        tandt_fps.append(ret['fps'])
        tandt_n_voxels.append(ret['n_voxels'])
    eval_path = f'{args.result_root}/{scene}/results.json'
    if os.path.exists(eval_path):
        with open(os.path.join(eval_path)) as f:
            ret = json.load(f)
            ret = ret[sorted(ret.keys())[-1]]
            tandt_psnr.append(ret['PSNR'])
            tandt_ssim.append(ret['SSIM'])
            tandt_lpips.append(ret['LPIPS'])
    else:
        tandt_psnr.append(tmp_psnr)
        tandt_ssim.append(0)
        tandt_lpips.append(0)

db_psnr = []
db_ssim = []
db_lpips = []
db_tr_time = []
db_fps = []
db_n_voxels = []
for scene in db_scenes:
    path = glob.glob(os.path.join(f'{args.result_root}/{scene}/test_stat/iter*.json'))
    if len(path) == 0:
        print(f'{scene:10s}: failed !!??')
        continue
    path = sorted(path)[-1]
    with open(path) as f:
        ret = json.load(f)
        tmp_psnr = ret['psnr']
        db_tr_time.append(ret['elapsed'] / 1000)
        db_fps.append(ret['fps'])
        db_n_voxels.append(ret['n_voxels'])
    eval_path = f'{args.result_root}/{scene}/results.json'
    if os.path.exists(eval_path):
        with open(os.path.join(eval_path)) as f:
            ret = json.load(f)
            ret = ret[sorted(ret.keys())[-1]]
            db_psnr.append(ret['PSNR'])
            db_ssim.append(ret['SSIM'])
            db_lpips.append(ret['LPIPS'])
    else:
        db_psnr.append(tmp_psnr)
        db_ssim.append(0)
        db_lpips.append(0)



def format_df_string(df):
    df = df.copy()
    df['scene'] = df['scene'].map(lambda s: s.rjust(15))
    df['psnr'] = df['psnr'].round(2)
    df['ssim'] = df['ssim'].round(3)
    df['lpips'] = df['lpips'].round(3)
    df['tr-mins'] = (df['tr-mins'] / 60).round(1)
    df['fps'] = df['fps'].round(1)
    df['#vox(M)'] = (df['#vox(M)'] / 1_000_000).round(1)
    return df.to_string(index=False)

def add_avg_row(df):
    df_avg = df.mean(axis=0, numeric_only=True).to_frame().transpose()
    df_avg['scene'] = 'AVG'
    return pd.concat([df, df_avg], ignore_index=True)

df_tandt = pd.DataFrame({
    'scene': tandt_scenes,
    'psnr': tandt_psnr,
    'ssim': tandt_ssim,
    'lpips': tandt_lpips,
    'tr-mins': tandt_tr_time,
    'fps': tandt_fps,
    '#vox(M)': tandt_n_voxels,
})
df_tandt = add_avg_row(df_tandt)

df_db = pd.DataFrame({
    'scene': db_scenes,
    'psnr': db_psnr,
    'ssim': db_ssim,
    'lpips': db_lpips,
    'tr-mins': db_tr_time,
    'fps': db_fps,
    '#vox(M)': db_n_voxels,
})
df_db = add_avg_row(df_db)

print(format_df_string(df_tandt))
print()
print(format_df_string(df_db))
