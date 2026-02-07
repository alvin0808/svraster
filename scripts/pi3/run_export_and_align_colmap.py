#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("[RUN]", " ".join(cmd))
    print("=" * 80, flush=True)
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Pi3X export (conf/normal/poses/ply) then align points to COLMAP GT in one shot."
    )

    # --- Export stage args ---
    p.add_argument("--data_root", required=True, type=str,
                   help="Image directory, e.g. /.../drawer_ex_3/images")
    p.add_argument("--model", required=True, type=str,
                   help="Path to model.safetensors (Pi3 weights).")
    p.add_argument("--out_dir", required=True, type=str,
                   help="Directory to save export outputs (camera_poses_i.pt, point_cloud_final_sor_filtered_i.ply, results_i.pt).")
    p.add_argument("--interval", type=int, default=2)
    p.add_argument("--conf_th", type=float, default=0.05)
    p.add_argument("--sor_k", type=int, default=50)
    p.add_argument("--sor_std", type=float, default=3.0)
    p.add_argument("--save_normal_npy", action="store_true")

    # --- Alignment stage args ---
    p.add_argument("--gt_bin", required=True, type=str,
                   help="Path to COLMAP images.bin (GT poses).")
    p.add_argument("--out_ply", required=True, type=str,
                   help="Output path for merged aligned point cloud, e.g. /.../sparse/0/aligned_points3D.ply")
    p.add_argument("--sample_rate", type=float, default=1.0)
    p.add_argument("--name_filter", type=str, default=None,
                   help="Optional: keep only GT images whose name contains this substring. Default: None.")

    # optional: allow choosing which scripts to call
    p.add_argument("--export_script", type=str, default="scripts/pi3/run_pi3x_export.py")
    p.add_argument("--align_script", type=str, default="scripts/pi3/colmap_pose_alignment.py")

    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]  # svraster/
    export_script = (repo_root / args.export_script).resolve()
    align_script = (repo_root / args.align_script).resolve()

    if not export_script.exists():
        raise FileNotFoundError(f"export_script not found: {export_script}")
    if not align_script.exists():
        raise FileNotFoundError(f"align_script not found: {align_script}")

    # 1) Export
    export_cmd = [
        sys.executable, str(export_script),
        "--data_root", args.data_root,
        "--out_dir", args.out_dir,
        "--interval", str(args.interval),
        "--conf_th", str(args.conf_th),
        "--sor_k", str(args.sor_k),
        "--sor_std", str(args.sor_std),
        "--model", args.model,
    ]
    if args.save_normal_npy:
        export_cmd.append("--save_normal_npy")

    run_cmd(export_cmd)

    # 2) Align
    align_cmd = [
        sys.executable, str(align_script),
        "--gt_bin", args.gt_bin,
        "--pi3_dir", args.out_dir,
        "--out_ply", args.out_ply,
        "--interval", str(args.interval),
        "--sample_rate", str(args.sample_rate),
    ]
    if args.name_filter is not None:
        align_cmd += ["--name_filter", args.name_filter]

    run_cmd(align_cmd)

    print("\n[OK] export + alignment finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
