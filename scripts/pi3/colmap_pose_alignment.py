#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, re, struct, collections, argparse
import numpy as np
import torch
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rscipy

# =========================== Utils ===========================
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)

def read_extrinsics_binary(path_to_model_file: str, name_filter: str | None):
    """
    Read COLMAP images.bin.
    If name_filter is not None, keep only images whose name contains name_filter.
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]

            name = ""
            c = read_next_bytes(fid, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = read_next_bytes(fid, 1, "c")[0]

            n2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * n2D, "ddq" * n2D)
            xys = np.column_stack([
                tuple(map(float, x_y_id_s[0::3])),
                tuple(map(float, x_y_id_s[1::3])),
            ])
            pids = np.array(tuple(map(int, x_y_id_s[2::3])))

            if (name_filter is None) or (name_filter in name):
                images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, pids)
    return images

def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _qvec2rotmat(q):  # [qw,qx,qy,qz]
    qw, qx, qy, qz = q
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    if n > 0:
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R

def images_bin_to_c2w_list(images_dict, order="name"):
    recs = []
    for img_id, I in images_dict.items():
        q, t = I.qvec.astype(np.float64), I.tvec.astype(np.float64)
        R_wc = _qvec2rotmat(q)  # w2c
        R_cw = R_wc.T
        t_cw = -R_cw @ t
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R_cw
        M[:3, 3] = t_cw
        recs.append((img_id, I.name, M))
    recs.sort(key=(lambda x: _natural_key(x[1])) if order == "name" else (lambda x: x[0]))
    return [r[2] for r in recs]

def load_pi3_c2w_opencv(pi3_path: str):
    # NOTE: weights_only=True works only if saved in a compatible way.
    # If you ever hit errors, drop weights_only.
    poses = torch.load(pi3_path, map_location="cpu", weights_only=True)
    arr = poses if torch.is_tensor(poses) else poses.get("poses")
    if arr is None:
        raise ValueError(f"Unrecognized PI3 poses format at {pi3_path}")
    if arr.dim() == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)  # (N,4,4)
    arr = arr.cpu().numpy().astype(np.float64)
    assert arr.shape[-2:] == (4, 4)
    return [arr[i] for i in range(arr.shape[0])]

def CFU_from_c2w_list_opencv(c2w_list):
    N = len(c2w_list)
    C = np.zeros((3, N)); F = np.zeros((3, N)); U = np.zeros((3, N))
    for i, M in enumerate(c2w_list):
        R = M[:3, :3]; t = M[:3, 3]
        C[:, i] = t
        f = R[:, 2]; f /= (np.linalg.norm(f) + 1e-12)
        u = R[:, 1]; u /= (np.linalg.norm(u) + 1e-12)
        F[:, i] = f; U[:, i] = u
    return C, F, U

def umeyama(X, Y):
    muX = X.mean(axis=1, keepdims=True); muY = Y.mean(axis=1, keepdims=True)
    X0, Y0 = X - muX, Y - muY
    Sigma = (Y0 @ X0.T) / X.shape[1]
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    varX = np.sum(np.sum(X0**2, axis=0)) / X.shape[1]
    s = np.trace(np.diag(D) @ S) / (varX + 1e-12)
    t = muY - s * R @ muX
    return s, R, t

def refine_similarity_opencv(
    pi3_list,
    C_gt_3xN, F_gt_3xN, U_gt_3xN,
    s0, R0, t0,
    lambda_f=2.0, lambda_u=1.0, w_pos=1.0,
    max_nfev=500
):
    N = min(len(pi3_list), C_gt_3xN.shape[1])
    C_p3 = np.stack([pi3_list[i][:3, 3] for i in range(N)], axis=0)
    F_p3 = np.stack([(pi3_list[i][:3, 2] / (np.linalg.norm(pi3_list[i][:3, 2]) + 1e-12)) for i in range(N)], axis=0)
    U_p3 = np.stack([(pi3_list[i][:3, 1] / (np.linalg.norm(pi3_list[i][:3, 1]) + 1e-12)) for i in range(N)], axis=0)

    C_gt = C_gt_3xN[:, :N].T
    F_gt = F_gt_3xN[:, :N].T
    U_gt = U_gt_3xN[:, :N].T

    rot0 = Rscipy.from_matrix(R0).as_rotvec()
    t0v = np.array(t0).reshape(3)
    log_s0 = np.log(max(1e-8, float(s0)))
    x0 = np.concatenate([rot0, t0v, [log_s0]])

    def residuals(x):
        rotvec = x[:3]; t = x[3:6]; s = np.exp(x[6])
        Rm = Rscipy.from_rotvec(rotvec).as_matrix()
        C_pred = (s * (C_p3 @ Rm.T)) + t
        r_pos = (C_pred - C_gt) * w_pos
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
    rot_ref = res.x[:3]; t_ref = res.x[3:6]; s_ref = float(np.exp(res.x[6]))
    R_ref = Rscipy.from_rotvec(rot_ref).as_matrix()
    return s_ref, R_ref, t_ref

def read_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)

def write_ply_points(points, path, binary=True, compressed=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    points = np.asarray(points, dtype=np.float64, order="C")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd, write_ascii=not binary, compressed=compressed)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_bin", required=True, help="Path to COLMAP images.bin (GT poses).")
    p.add_argument("--pi3_dir", required=True, help="Directory containing camera_poses_{i}.pt and point_cloud_final_sor_filtered_{i}.ply")
    p.add_argument("--out_ply", required=True, help="Output .ply path for merged aligned points.")
    p.add_argument("--name_filter", default=None, help="Keep only GT images whose filename contains this substring.")
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

    images = read_extrinsics_binary(args.gt_bin, args.name_filter)
    gt_all = images_bin_to_c2w_list(images, order="name")
    if len(gt_all) == 0:
        print("[ERROR] Empty GT (after filtering).", file=sys.stderr)
        sys.exit(1)

    pi3_dir = args.pi3_dir
    interval = int(args.interval)
    sample_rate = float(args.sample_rate)
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("--sample_rate must be in (0, 1].")

    merged = []
    M = len(gt_all)

    for i in range(interval):
        pi3_path = os.path.join(pi3_dir, args.poses_pat.format(i=i))
        ply_path = os.path.join(pi3_dir, args.ply_pat.format(i=i))

        if not (os.path.isfile(pi3_path) and os.path.isfile(ply_path)):
            print(f"[WARN] skip i={i}: missing files: {pi3_path} or {ply_path}")
            continue

        pi3_list = load_pi3_c2w_opencv(pi3_path)
        N = len(pi3_list)
        if N == 0:
            print(f"[WARN] i={i}: empty Pi3 poses")
            continue

        idxs = [i + k*interval for k in range(N) if (i + k*interval) < M]
        if len(idxs) == 0:
            print(f"[WARN] i={i}: no GT indices")
            continue
        if len(idxs) < N:
            pi3_list = pi3_list[:len(idxs)]
            N = len(pi3_list)

        gt_sel = [gt_all[j] for j in idxs]
        C_gt, F_gt, U_gt = CFU_from_c2w_list_opencv(gt_sel)

        C_pi3 = np.stack([Mat[:3, 3] for Mat in pi3_list], axis=1)  # 3xN
        s0, R0, t0 = umeyama(C_pi3, C_gt)

        C_align0 = (s0 * (R0 @ C_pi3)) + t0.reshape(3, 1)
        rmse0 = np.sqrt(np.mean(np.sum((C_align0 - C_gt)**2, axis=0)))
        print(f"[i={i}] init similarity: scale={s0:.6f}, centers RMSE init={rmse0:.6e}")

        s, R, t = refine_similarity_opencv(
            pi3_list, C_gt, F_gt, U_gt, s0, R0, t0,
            lambda_f=args.lambda_f, lambda_u=args.lambda_u, w_pos=args.w_pos,
            max_nfev=args.max_nfev
        )
        C_align = (s * (R @ C_pi3)) + t.reshape(3, 1)
        rmse = np.sqrt(np.mean(np.sum((C_align - C_gt)**2, axis=0)))
        print(f"[i={i}] centers RMSE refine={rmse:.6e}")

        pts = read_ply_points(ply_path)
        if pts.size == 0:
            print(f"[WARN] i={i}: empty points")
            continue

        pts_aligned = (s * (R @ pts.T)).T + t.reshape(1, 3)

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
