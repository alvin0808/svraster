#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, re, argparse
import numpy as np
import torch
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rscipy
import json

# =========================== Utils ===========================
def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def umeyama(X, Y):
    """X->Y similarity (s,R,t). X,Y: 3xN"""
    muX = X.mean(axis=1, keepdims=True); muY = Y.mean(axis=1, keepdims=True)
    X0, Y0 = X - muX, Y - muY
    Sigma = (Y0 @ X0.T) / X.shape[1]
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    Rm = U @ S @ Vt
    varX = np.sum(np.sum(X0**2, axis=0)) / X.shape[1]
    s = np.trace(np.diag(D) @ S) / (varX + 1e-12)
    t = muY - s * Rm @ muX
    return s, Rm, t

def load_gt_json_opengl(gt_json_path: str, name_filter: str | None):
    """
    transforms_train.json (NeRF)에서 GT OpenGL c2w를 읽고,
    center/forward/up을 반환한다.

    - transform_matrix: OpenGL c2w
    - forward(OpenGL): -Z
    - up(OpenGL): +Y
    """
    with open(gt_json_path, "r") as f:
        gt = json.load(f)

    frames = gt.get("frames", [])
    if len(frames) == 0:
        return [], None, None, None

    # 필터링 + 정렬: name( file_path ) 기준 natural sort
    recs = []
    for fr in frames:
        # NeRF json은 보통 file_path를 갖고 있음
        name = fr.get("file_path", fr.get("name", ""))
        if (name_filter is None) or (name_filter in name):
            M = np.array(fr["transform_matrix"], dtype=np.float64)  # OpenGL c2w
            recs.append((name, M))
    recs.sort(key=lambda x: _natural_key(x[0]))

    if len(recs) == 0:
        return [], None, None, None

    centers, forwards, ups = [], [], []
    c2w_list = []
    for name, M in recs:
        c2w_list.append(M)
        centers.append(M[:3, 3])
        fwd = -M[:3, 2]; fwd /= (np.linalg.norm(fwd) + 1e-12)  # forward = -Z
        up  =  M[:3, 1]; up  /= (np.linalg.norm(up ) + 1e-12)  # up = +Y
        forwards.append(fwd); ups.append(up)

    C = np.stack(centers,  axis=1)  # 3xN
    F = np.stack(forwards, axis=1)  # 3xN
    U = np.stack(ups,      axis=1)  # 3xN
    return c2w_list, C, F, U

def load_pi3_c2w_opencv(pi3_path: str):
    """
    Pi3 포즈 로드: (N,4,4) 또는 (1,N,4,4) 형태 지원.
    Pi3는 c2w(OpenCV)로 가정.
    """
    poses = torch.load(pi3_path, map_location="cpu", weights_only=True)
    arr = poses if torch.is_tensor(poses) else poses.get("poses")
    if arr is None:
        raise ValueError(f"Unrecognized PI3 poses format at {pi3_path}")
    if arr.dim() == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)  # (N,4,4)
    arr = arr.cpu().numpy().astype(np.float64)
    assert arr.shape[-2:] == (4, 4)
    return [arr[i] for i in range(arr.shape[0])]

def opencv_c2w_list_to_opengl(c2w_cv_list):
    """
    c2w(OpenCV) -> c2w(OpenGL)
    (네 예시처럼) 오른쪽에 diag([1,-1,-1,1]) 곱한다.
    """
    T_cv2gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    out = []
    for M in c2w_cv_list:
        out.append(M @ T_cv2gl)
    return out

def read_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)

def write_ply_points(points, path, binary=True, compressed=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    points = np.asarray(points, dtype=np.float64, order="C")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd, write_ascii=not binary, compressed=compressed)

def refine_similarity_opengl(
    pi3_gl_list,
    C_gt_3xN, F_gt_3xN, U_gt_3xN,
    s0, R0, t0,
    lambda_f=2.0, lambda_u=1.0, w_pos=1.0,
    max_nfev=500
):
    """
    OpenGL 좌표계에서 위치+방향(forward/up)을 같이 맞추는 similarity refine.
    반환 (s,R,t)는 아래 형태로 쓰인다:

      x_gl = s * (R @ x_pi3_gl) + t

    즉, pts_aligned = (s * (R @ pts.T)).T + t
    """
    N = min(len(pi3_gl_list), C_gt_3xN.shape[1])
    if N <= 0:
        raise ValueError("No overlapping frames for refine.")

    # Pi3(OpenGL) center/forward/up
    C_p3 = np.stack([pi3_gl_list[i][:3, 3] for i in range(N)], axis=0)  # (N,3)

    F_p3 = np.stack([
        (-pi3_gl_list[i][:3, 2] / (np.linalg.norm(pi3_gl_list[i][:3, 2]) + 1e-12))
        for i in range(N)
    ], axis=0)

    U_p3 = np.stack([
        ( pi3_gl_list[i][:3, 1] / (np.linalg.norm(pi3_gl_list[i][:3, 1]) + 1e-12))
        for i in range(N)
    ], axis=0)

    # GT(OpenGL)
    C_gt = C_gt_3xN[:, :N].T
    F_gt = F_gt_3xN[:, :N].T
    U_gt = U_gt_3xN[:, :N].T

    # init params
    rot0 = Rscipy.from_matrix(R0).as_rotvec()
    t0v  = np.array(t0).reshape(3)
    log_s0 = np.log(max(1e-8, float(s0)))
    x0 = np.concatenate([rot0, t0v, [log_s0]])

    def residuals(x):
        rotvec = x[:3]; t = x[3:6]; s = np.exp(x[6])
        Rm = Rscipy.from_rotvec(rotvec).as_matrix()

        # positions
        C_pred = (s * (C_p3 @ Rm.T)) + t
        r_pos = (C_pred - C_gt) * w_pos

        # directions
        F_pred = (F_p3 @ Rm.T); F_pred /= (np.linalg.norm(F_pred, axis=1, keepdims=True) + 1e-12)
        U_pred = (U_p3 @ Rm.T); U_pred /= (np.linalg.norm(U_pred, axis=1, keepdims=True) + 1e-12)
        r_f = (F_pred - F_gt) * np.sqrt(lambda_f)
        r_u = (U_pred - U_gt) * np.sqrt(lambda_u)

        return np.concatenate([r_pos.ravel(), r_f.ravel(), r_u.ravel()], axis=0)

    lower = np.array([-np.inf]*6 + [np.log(1e-6)])
    upper = np.array([ np.inf]*6 + [np.log(1e+6)])
    res = least_squares(
        residuals, x0, method="trf",
        loss="soft_l1", f_scale=0.05,
        max_nfev=max_nfev, bounds=(lower, upper)
    )

    rot_ref = res.x[:3]
    t_ref   = res.x[3:6]
    s_ref   = float(np.exp(res.x[6]))
    R_ref   = Rscipy.from_rotvec(rot_ref).as_matrix()
    return s_ref, R_ref, t_ref, res

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_json", required=True, help="Path to transforms_train.json (GT OpenGL c2w).")
    p.add_argument("--pi3_dir", required=True, help="Directory containing Pi3 outputs.")
    p.add_argument("--out_ply", required=True, help="Output .ply path for merged aligned points.")
    p.add_argument("--name_filter", default=None, help="Keep only GT frames whose file_path contains this substring.")
    p.add_argument("--interval", type=int, default=1, help="INTERVAL used when sampling GT indices: i + k*interval")
    p.add_argument("--sample_rate", type=float, default=1.0, help="Random keep probability per point after alignment (0~1].")
    p.add_argument("--poses_pat", default="camera_poses_{i}.pt", help="Filename pattern inside pi3_dir.")
    p.add_argument("--ply_pat", default="point_cloud_final_sor_filtered_{i}.ply", help="Filename pattern inside pi3_dir.")
    p.add_argument("--lambda_f", type=float, default=2.0)
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--w_pos", type=float, default=1.0)
    p.add_argument("--max_nfev", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

# =========================== Main ===========================
def main():
    args = parse_args()
    np.random.seed(args.seed)

    gt_list, C_gt_all, F_gt_all, U_gt_all = load_gt_json_opengl(args.gt_json, args.name_filter)
    if len(gt_list) == 0:
        print("[ERROR] Empty GT (after filtering).", file=sys.stderr)
        sys.exit(1)

    interval = int(args.interval)
    sample_rate = float(args.sample_rate)
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("--sample_rate must be in (0, 1].")

    merged = []
    M = C_gt_all.shape[1]
    pi3_dir = args.pi3_dir

    for i in range(interval):
        pi3_path = os.path.join(pi3_dir, args.poses_pat.format(i=i))
        ply_path = os.path.join(pi3_dir, args.ply_pat.format(i=i))

        if not (os.path.isfile(pi3_path) and os.path.isfile(ply_path)):
            print(f"[WARN] skip i={i}: missing files: {pi3_path} or {ply_path}")
            continue

        # Pi3: c2w(OpenCV) -> c2w(OpenGL)
        pi3_cv = load_pi3_c2w_opencv(pi3_path)
        if len(pi3_cv) == 0:
            print(f"[WARN] i={i}: empty Pi3 poses")
            continue
        pi3_gl = opencv_c2w_list_to_opengl(pi3_cv)

        N = len(pi3_gl)
        idxs = [i + k*interval for k in range(N) if (i + k*interval) < M]
        if len(idxs) == 0:
            print(f"[WARN] i={i}: no GT indices")
            continue
        if len(idxs) < N:
            pi3_gl = pi3_gl[:len(idxs)]
            N = len(pi3_gl)

        # GT subset
        C_gt = C_gt_all[:, idxs]  # 3xN
        F_gt = F_gt_all[:, idxs]
        U_gt = U_gt_all[:, idxs]

        # Umeyama init (centers only)
        C_pi3 = np.stack([pi3_gl[k][:3, 3] for k in range(N)], axis=1)  # 3xN
        s0, R0, t0 = umeyama(C_pi3, C_gt)

        C_align0 = (s0 * (R0 @ C_pi3)) + t0.reshape(3, 1)
        rmse0 = np.sqrt(np.mean(np.sum((C_align0 - C_gt)**2, axis=0)))
        print(f"[i={i}] init similarity: scale={s0:.6f}, centers RMSE init={rmse0:.6e}")

        # Refine with forward/up
        s, Rm, t, res = refine_similarity_opengl(
            pi3_gl, C_gt, F_gt, U_gt, s0, R0, t0,
            lambda_f=args.lambda_f, lambda_u=args.lambda_u, w_pos=args.w_pos,
            max_nfev=args.max_nfev
        )

        C_align = (s * (Rm @ C_pi3)) + t.reshape(3, 1)
        rmse = np.sqrt(np.mean(np.sum((C_align - C_gt)**2, axis=0)))
        print(f"[i={i}] refine: status={res.status}, cost={res.cost:.6f}, scale={s:.6f}, centers RMSE refine={rmse:.6e}")

     
        pts = read_ply_points(ply_path)
        if pts.size == 0:
            print(f"[WARN] i={i}: empty points")
            continue

        # 최종 similarity 적용 (OpenGL)
        pts_aligned = (s * (Rm @ pts.T)).T + t.reshape(1, 3)

        Mpts = pts_aligned.shape[0]
        if sample_rate >= 1.0:
            pts_s = pts_aligned
        else:
            mask = (np.random.rand(Mpts) < sample_rate)
            if not mask.any():
                mask[np.random.randint(0, Mpts)] = True
            pts_s = pts_aligned[mask]

        merged.append(pts_s)
        print(f"[i={i}] {Mpts} -> {pts_s.shape[0]} sampled")

    if len(merged) == 0:
        print("[ERROR] nothing merged", file=sys.stderr)
        sys.exit(1)

    merged = np.concatenate(merged, axis=0)
    write_ply_points(merged, args.out_ply, binary=True, compressed=False)
    print(f"[OK] saved merged aligned points -> {args.out_ply}  (total {merged.shape[0]})")

if __name__ == "__main__":
    main()
