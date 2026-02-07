from __future__ import annotations

import os, sys
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]
PI3_ROOT  = REPO_ROOT / "third_party" / "pi3"
sys.path.insert(0, str(PI3_ROOT))

import torch
import open3d as o3d
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from safetensors.torch import load_file
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root", type=str, required=True,
        help="Directory that contains the input images (e.g., .../drawer_ex_3/images). conf/normal will be saved under its parent directory."
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to model.safetensors"
    )
    p.add_argument(
        "--out_dir", type=str, default=str(REPO_ROOT / "output" / "pi3_export"),
        help="Directory for ply/pt outputs (NOT conf/normal)."
    )
    p.add_argument("--interval", type=int, default=6)
    p.add_argument("--conf_th", type=float, default=0.05)
    p.add_argument("--sor_k", type=int, default=50)
    p.add_argument("--sor_std", type=float, default=3.0)
    p.add_argument(
        "--save_normal_npy", action="store_true",
        help="Also save normal_XXX.npy (float32) alongside normal PNGs."
    )
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    # conf/normal은 train에서 쓰므로 데이터 폴더 아래에 저장
    export_root = data_root.parent

    if not data_root.exists():
        raise FileNotFoundError(f"--data_root does not exist: {data_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"--model does not exist: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (export_root / "conf").mkdir(parents=True, exist_ok=True)
    (export_root / "normal").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Pi3().to(device).eval()

    weight = load_file(str(model_path))
    model.load_state_dict(weight)

    interval = args.interval

    # mixed precision dtype
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float32

    def normals_from_points_4nb(P: torch.Tensor, valid: torch.Tensor, eps: float = 1e-8):
        # P: (H,W,3), valid: (H,W) bool
        Pr = torch.roll(P, shifts=-1, dims=1)  # right neighbor
        Pd = torch.roll(P, shifts=-1, dims=0)  # down neighbor
        du = Pr - P
        dv = Pd - P
        N = torch.cross(dv, du, dim=-1)

        v = valid.clone()
        v[-1, :] = False
        v[:, -1] = False

        Nnorm = torch.linalg.norm(N, dim=-1, keepdim=True).clamp_min(eps)
        Nunit = torch.where(v[..., None], N / Nnorm, torch.zeros_like(N))
        return Nunit, v

    for i in range(interval):
        imgs = load_images_as_tensor(str(data_root), interval=interval, start_num=i).to(device)

        print(f"[i={i}] Running model inference... imgs={tuple(imgs.shape)}")
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    results = model(imgs[None])
            else:
                results = model(imgs[None])

        print(f"[i={i}] Reconstruction complete!")

        # ---------------- conf map 저장 ----------------
        if "conf" not in results:
            raise KeyError("results['conf'] not found.")

        conf_raw = results["conf"]                      # (B,N,H,W) or (B,N,H,W,1)
        conf_sig = torch.sigmoid(conf_raw)

        if conf_sig.dim() == 5 and conf_sig.shape[-1] == 1:
            conf_sig = conf_sig[..., 0]                 # -> (B,N,H,W)

        conf_sig = conf_sig.to(torch.float32)

        conf_dir = export_root / "conf"
        B_conf, N_conf = conf_sig.shape[:2]
        if B_conf != 1:
            raise ValueError(f"Expected batch size 1 for conf, got {B_conf}")

        for n in range(N_conf):
            conf_map = conf_sig[0, n]                   # (H,W)
            conf_img = (conf_map.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
            image_num = n * interval + i
            out_conf = conf_dir / f"{image_num:03d}.png"
            Image.fromarray(conf_img, mode="L").save(out_conf)

        # ---------------- point cloud 저장 ----------------
        points_tensor = results["points"]
        conf_threshold = args.conf_th

        # 기존 가정 유지: conf mask와 points 텐서가 broadcast/indexing 가능
        mask = torch.sigmoid(results["conf"]) > conf_threshold
        filtered_points = points_tensor[mask.expand_as(points_tensor)]
        filtered_points_np = filtered_points.reshape(-1, 3).detach().cpu().numpy()
        filtered_points_np = np.asarray(filtered_points_np, dtype=np.float64, order="C")
        if filtered_points_np.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points_np)

            #ply_path_conf = out_dir / f"point_cloud_conf_filtered_{i}.ply"
            #o3d.io.write_point_cloud(str(ply_path_conf), pcd)
            #print(f"[i={i}] conf-filtered ply saved: {ply_path_conf} ({len(pcd.points)} pts)")

            print(f"[i={i}] Applying SOR...")
            pcd_sor, _ = pcd.remove_statistical_outlier(
                nb_neighbors=args.sor_k,
                std_ratio=args.sor_std
            )
            ply_path_sor = out_dir / f"point_cloud_final_sor_filtered_{i}.ply"
            o3d.io.write_point_cloud(str(ply_path_sor), pcd_sor)
            print(f"[i={i}] SOR ply saved: {ply_path_sor} ({len(pcd_sor.points)} pts)")
        else:
            print(f"[i={i}] Warning: No points above conf_th={conf_threshold}")

        # ---------------- results/poses 저장 ----------------
        results_pt_path = out_dir / f"results_{i}.pt"
        torch.save(results, str(results_pt_path))

        poses_path = out_dir / f"camera_poses_{i}.pt"
        torch.save(results["camera_poses"], str(poses_path))

        # ---------------- normal map 저장 ----------------
        if "local_points" not in results:
            raise KeyError("results['local_points'] not found.")

        lp = results["local_points"].to(torch.float32)  # (B,N,H,W,3)
        B, N, H, W, _ = lp.shape
        if B != 1:
            raise ValueError(f"Expected batch size 1 for local_points, got {B}")

        normals_dir = export_root / "normal"

        for n in range(N):
            P0 = lp[0, n]                                # (H,W,3)
            valid = (P0[..., 2] > 0)

            Nunit, vmask = normals_from_points_4nb(P0, valid, eps=1e-8)

            Ncpu = Nunit.clamp(-1, 1).cpu().numpy()      # float32-ish
            vis = ((Ncpu + 1.0) * 0.5 * 255.0).astype(np.uint8)
            vis[~vmask.cpu().numpy()] = 0

            image_num = n * interval + i
            out_png = normals_dir / f"{image_num:03d}.png"
            Image.fromarray(vis, mode="RGB").save(out_png)

            if args.save_normal_npy:
                out_npy = normals_dir / f"normal_{image_num:03d}.npy"
                np.save(out_npy, Ncpu.astype(np.float32))

        print(f"[i={i}] Done.")

if __name__ == "__main__":
    main()
