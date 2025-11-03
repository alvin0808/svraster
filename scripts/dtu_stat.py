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

scenes = [
    'scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122'
]

cf = []
tr_time = []
fps = []
n_voxels = []

for scene in scenes:
    eval_path = sorted(glob.glob(f'{args.result_root}/{scene}/test_stat/iter*.json'))
    if len(eval_path):
        eval_path = eval_path[-1]
        with open(eval_path) as f:
            ret = json.load(f)
            tr_time.append(ret['elapsed'] / 1000)
            n_voxels.append(ret['n_voxels'])
    else:
        tr_time.append(0)
        n_voxels.append(0)

    eval_path = sorted(glob.glob(f'{args.result_root}/{scene}/train/*.txt'))
    if len(eval_path):
        eval_path = eval_path[-1]
        with open(eval_path) as f:
            fps.append(float([line.strip().split('=')[1] for line in f if line.startswith('fps')][-1]))
    else:
        fps.append(0)

    eval_path = f'{args.result_root}/{scene}/mesh/latest/mesh_dense_cleaned_for_eval.ply.json'
    if os.path.isfile(eval_path):
        with open(eval_path) as f:
            ret = json.load(f)
            cf.append(ret['overall'])
    else:
        cf.append(10)



def format_df_string(df):
    df = df.copy()
    df['scene'] = df['scene'].map(lambda s: s.rjust(15))
    df['cf-dist'] = df['cf-dist'].round(2)
    df['tr-mins'] = (df['tr-mins'] / 60).round(1)
    df['fps'] = df['fps'].round(1)
    df['#vox(M)'] = (df['#vox(M)'] / 1_000_000).round(1)
    return df.to_string(index=False)

def add_avg_row(df):
    df_avg = df.mean(axis=0, numeric_only=True).to_frame().transpose()
    df_avg['scene'] = 'AVG'
    return pd.concat([df, df_avg], ignore_index=True)

df = pd.DataFrame({
    'scene': scenes,
    'cf-dist': cf,
    'tr-mins': tr_time,
    'fps': fps,
    '#vox(M)': n_voxels,
})
df = add_avg_row(df)

print(format_df_string(df))

