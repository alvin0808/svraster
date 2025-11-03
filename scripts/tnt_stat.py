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
parser.add_argument('--suffix', default='_r2.0')
args = parser.parse_args()

all_scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']

all_fscore = []
all_precision = []
all_recall = []
all_tr_time = []
all_fps = []
all_n_voxels = []
for scene in all_scenes:
    path = glob.glob(os.path.join(f'{args.result_root}/{scene}/test_stat/iter*.json'))
    if len(path) == 0:
        print(f'{scene:10s}: failed !!??')
        continue
    path = sorted(path)[-1]
    with open(path) as f:
        ret = json.load(f)
        all_tr_time.append(ret['elapsed'] / 1000)
        # all_fps.append(ret['fps'])
        all_n_voxels.append(ret['n_voxels'])
    n_iter = int(os.path.split(path)[1].replace('iter', '').replace('.json', ''))
    fps_path = f'{args.result_root}/{scene}/train/ours_{n_iter}{args.suffix}.txt'
    with open(fps_path) as f:
        fps = float(f.read().strip().split()[-1].split('=')[1])
        all_fps.append(fps)
    eval_path = f'{args.result_root}/{scene}/mesh/latest/evaluation/result.json'
    if os.path.exists(eval_path):
        with open(os.path.join(eval_path)) as f:
            ret = json.load(f)
            all_fscore.append(ret['f-score'])
            all_precision.append(ret['precision'])
            all_recall.append(ret['recall'])
    else:
        all_fscore.append(0)
        all_precision.append(0)
        all_recall.append(0)



def format_df_string(df):
    df = df.copy()
    df['scene'] = df['scene'].map(lambda s: s.rjust(15))
    df['f-score'] = df['f-score'].round(2)
    df['prec.'] = df['prec.'].round(2)
    df['recall'] = df['recall'].round(2)
    df['tr. mins'] = (df['tr. mins'] / 60).round(1)
    df['fps'] = df['fps'].round(1)
    df['#vox (M)'] = (df['#vox (M)'] / 1_000_000).round(1)
    return df.to_string()

def add_avg_row(df):
    df_avg = df.mean(axis=0, numeric_only=True).to_frame().transpose()
    df_avg['scene'] = 'AVG'
    return pd.concat([df, df_avg], ignore_index=True)

df = pd.DataFrame({
    'scene': all_scenes,
    'f-score': all_fscore,
    'prec.': all_precision,
    'recall': all_recall,
    'tr. mins': all_tr_time,
    'fps': all_fps,
    '#vox (M)': all_n_voxels,
})
df = add_avg_row(df)

print(format_df_string(df))
