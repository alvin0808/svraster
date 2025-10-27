# src/utils/sdf_init_utils.py

import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def initialize_sdf_from_sfm(sfm_init, cameras, grid_pts_key, scene_center, scene_extent, voxel_size, debug_ply_output_path):
    """
    SfM 데이터를 사용하여 SDF(_geo_grid_pts) 텐서를 초기화하는 전체 과정을 수행합니다.
    (카메라 리스트 순서가 COLMAP IMAGE_ID와 일치한다고 가정)
    """
    print("loaded point num:", len(sfm_init.points_xyz))
    
    # --- 1. DBSCAN을 이용한 동적 노이즈 필터링 ---
    # (이전과 동일)
    '''
    print("Step 1/5: Filtering SfM points with DBSCAN...")
    eps = voxel_size.item() * 4.0
    min_samples = 10
    print(f"Using Voxel Size: {voxel_size.item():.4f} for DBSCAN eps: {eps:.4f}")
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(sfm_init.points_xyz)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) > 0:
        largest_cluster_label = unique_labels[counts.argmax()]
        inlier_mask = (labels == largest_cluster_label)
        filtered_xyz = sfm_init.points_xyz[inlier_mask]
        
        print(f"Filtered {len(sfm_init.points_xyz)} -> {len(filtered_xyz)} points.")
        debug_ply_path = debug_ply_output_path
        if debug_ply_path is not None:
            # ==========================================================
            # === plyfile을 사용하는 코드 (RGB 정보 제외) ===
            # ==========================================================
            try:
                print(f"Saving {len(filtered_xyz)} filtered points (XYZ only) to {debug_ply_path} using plyfile...")
                from plyfile import PlyData, PlyElement

                # plyfile이 요구하는 데이터 형식 (XYZ만)으로 변환합니다.
                # (x, y, z) 필드를 갖는 structured array를 만듭니다.
                vertices = np.core.records.fromarrays(
                    filtered_xyz.transpose(), 
                    names='x, y, z',
                    formats='f4, f4, f4'
                )

                el = PlyElement.describe(vertices, 'vertex')
                
                # 파일을 바이너리 쓰기 모드로 엽니다.
                with open(str(debug_ply_path), 'wb') as f:
                    PlyData([el]).write(f)

                print(">>> Successfully saved PLY file (XYZ only) with plyfile. <<<")

            except Exception as e:
                print(f"!!! An error occurred while saving with plyfile: {e} !!!")
            # ==========================================================
        # ==========================================================
    else:
        print("Warning: DBSCAN found no clusters. Using original points.")
        filtered_xyz = sfm_init.points_xyz
        #filtered_idx_to_pid = sfm_init.index_to_point_id
        #filtered_pid_to_img_ids = sfm_init.point_id_to_image_ids
    '''
    #filtered_xyz= sfm_init.points_xyz #will not use filtering for now
    
    sampling_rate = 0.01
    print("step 1/4: Downsampling SfM points for SDF initialization...")
    num_points = len(sfm_init.points_xyz)
    sample_indices = np.random.choice(num_points, size=int(num_points * sampling_rate), replace=False)
    filtered_xyz = sfm_init.points_xyz[sample_indices]
    if debug_ply_output_path is not None:
            # ==========================================================
            # === plyfile을 사용하는 코드 (RGB 정보 제외) ===
            # ==========================================================
            try:
                print(f"Saving {len(filtered_xyz)} filtered points (XYZ only) to {debug_ply_output_path} using plyfile...")
                from plyfile import PlyData, PlyElement

                # plyfile이 요구하는 데이터 형식 (XYZ만)으로 변환합니다.
                # (x, y, z) 필드를 갖는 structured array를 만듭니다.
                vertices = np.core.records.fromarrays(
                    filtered_xyz.transpose(), 
                    names='x, y, z',
                    formats='f4, f4, f4'
                )

                el = PlyElement.describe(vertices, 'vertex')
                
                # 파일을 바이너리 쓰기 모드로 엽니다.
                with open(str(debug_ply_output_path), 'wb') as f:
                    PlyData([el]).write(f)

                print(">>> Successfully saved PLY file (XYZ only) with plyfile. <<<")

            except Exception as e:
                print(f"!!! An error occurred while saving with plyfile: {e} !!!")
    # --- 3. K-D Tree 생성 ---
    print("Step 1/4: Building K-D Tree...")
    kdtree = KDTree(filtered_xyz)
    
    # ==========================================================
    # === 3. SDF 값 계산 (threshold_dist 로직 복원) ===
    # ==========================================================
    print("Step 2/4: Calculating SDF values (axis-aligned cube 0-set + negative fill)...")
    level = 16
    grid_pts_pos = grid_pts_key.float() / (1 << level)
    grid_pts_pos = grid_pts_pos * scene_extent + (scene_center - 0.5 * scene_extent)
    query_points_np = grid_pts_pos.cpu().numpy()   # (N,3)

    N = len(query_points_np)
    sdf_values = np.empty(N, dtype=np.float32)

    tau0 = 0.5 * float(voxel_size.item())         

    dist_all, nn_idx = kdtree.query(query_points_np, k=1)

    sdf_values[:] = -dist_all
    '''
    nn_pts = filtered_xyz[nn_idx]                
    abs_diff = np.abs(query_points_np - nn_pts)    # (N,3)
    inside_cube = (abs_diff[:, 0] <= tau0) & (abs_diff[:, 1] <= tau0) & (abs_diff[:, 2] <= tau0)

    sdf_values[inside_cube] = 0.0
    '''

    

    # 한 개만 샘플로 확인
    cams_meta, report = validate_and_prepare_cameras(cameras)  # cameras는 DataPack.get_train_cameras()의 반환값
    #print(report)
    ##print("Sample K/R/t:")
    #print(cams_meta[0]["K"])
    #print(cams_meta[0]["R_wc"])
    #print(cams_meta[0]["t_wc"])
    zero_mask_t = torch.from_numpy((sdf_values == 0.0)).bool()  # CPU 텐서로 만들 예정
    grid_xyz_cpu = grid_pts_pos.detach().cpu().float()
    sdf_cpu = torch.from_numpy(sdf_values).float()  # (N,)

    # 카메라 메타는 validate_and_prepare_cameras 에서 이미 CPU 텐서로 확보한 상태 (cams_meta)
    D0_list = _build_depth_maps_from_points(
        sfm_init.points_xyz,  # ← numpy (M,3), 위에서 다운샘플한 SfM 포인트
        cams_meta, bin_max=64, chunk=200_000
    )


    import math
    if len(D0_list) == 0:
        print("[D0 CHECK] No cameras in D0_list.")
    else:
        D0 = D0_list[0]  # 24번째 카메라
        cam0 = cams_meta[0]
        bin_H, bin_W = D0.shape
        print(f"[D0 CHECK] cam='{cam0['name']}'  D0.shape={tuple(D0.shape)} (bin_H, bin_W)")

        finite = torch.isfinite(D0)
        fill_ratio = finite.float().mean().item() * 100.0
        print(f"[D0 CHECK] finite ratio = {fill_ratio:.2f}% (filled bins)")

        if finite.any():
            zmin = D0[finite].min().item()
            zmax = D0[finite].max().item()
            print(f"[D0 CHECK] depth range (finite) = [{zmin:.6f}, {zmax:.6f}] (camera-Z)")

        # 상단 좌측 코너 일부(최대 8x10) 값 미리보기 (inf는 'inf'로 표시)
        max_r, max_c = min(48, bin_H), min(64, bin_W)
        print(f"[D0 CHECK] top-left {max_r}x{max_c} preview:")
        for r in range(max_r):
            row_vals = []
            for c in range(max_c):
                v = D0[r, c].item()
                row_vals.append("   inf" if not math.isfinite(v) else f"{v:6.3f}")
            print(" ".join(row_vals))



    sdf_cpu = _flip_sign_via_visibility_with_margin(
        grid_xyz_cpu, sdf_cpu, cams_meta, D0_list, tau0=float(voxel_size.item())*0.5,
        eps=1e-4, chunk=200_000
    )
        
    sdf_values = sdf_cpu.numpy().astype(np.float32).ravel()  # ← flip 결과 반영!
    sdf_values *= 2.0
    return torch.from_numpy(sdf_values).float().to("cuda").unsqueeze(1)





#uitils
# ---- sdf_init_utils.py 상단에 추가 ----
import math
import torch

def _fx_fy_from_fov(fovx, fovy, W, H):
    fx = (W * 0.5) / math.tan(fovx * 0.5)
    fy = (H * 0.5) / math.tan(fovy * 0.5)
    return fx, fy

def _K_from_params(fovx, fovy, W, H, cx_p, cy_p, device="cpu", dtype=torch.float32):
    fx, fy = _fx_fy_from_fov(fovx, fovy, W, H)
    cx = cx_p * (W - 1)
    cy = cy_p * (H - 1)
    return torch.tensor([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=dtype, device=device)

def _coerce_cameras_to_list(cameras):
    # CameraList는 __len__/__getitem__이 있어서 list()로 바로 풀린다
    if isinstance(cameras, (list, tuple)):
        return list(cameras)
    try:
        return list(cameras)
    except TypeError:
        # 혹시 모를 래퍼 속성들도 시도
        for attr in ("camera_list", "cameras", "list", "_cameras", "_list", "data"):
            if hasattr(cameras, attr):
                try:
                    return list(getattr(cameras, attr))
                except Exception:
                    pass
        raise ValueError(f"`cameras`를 리스트로 변환할 수 없음: type={type(cameras)}")

@torch.no_grad()
def validate_and_prepare_cameras(cameras):
    cams = _coerce_cameras_to_list(cameras)
    if len(cams) == 0:
        raise ValueError("cameras list is empty.")
    lines = []
    metas = []
    for i, cam in enumerate(cams):
        name = getattr(cam, "image_name", f"cam_{i}")
        w2c = cam.w2c.detach().float().cpu()
        c2w = cam.c2w.detach().float().cpu()
        img = cam.image            # device 상관 없음
        C,H,W = img.shape
        fovx = float(cam.fovx); fovy = float(cam.fovy)
        cx_p = float(getattr(cam, "cx_p", 0.5))
        cy_p = float(getattr(cam, "cy_p", 0.5))
        near = float(getattr(cam, "near", 0.02))
        # 역행렬 일관성
        err = torch.linalg.norm(w2c @ c2w - torch.eye(4), ord='fro').item()
        if err > 1e-3:
            lines.append(f"[{name}] WARN: w2c@c2w ≠ I (‖·‖F={err:.2e})")
        # K 구성
        K = _K_from_params(fovx, fovy, W, H, cx_p, cy_p, device="cpu")
        metas.append({
            "name": name,
            "R_wc": w2c[:3,:3].contiguous(),
            "t_wc": w2c[:3, 3].contiguous(),
            "K": K, "W": W, "H": H, "near": near,
            "cx_p": cx_p, "cy_p": cy_p
        })
        lines.append(f"[{name}] OK  W×H={W}×{H}  fovx={fovx:.3f} fovy={fovy:.3f}  cxp/cyp={cx_p:.2f}/{cy_p:.2f}")
    return metas, "\n".join(lines)


# ========= 3) 0-셋 기반 저해상도 Z-buffer 생성 =========
import math
import torch

@torch.no_grad()
def _project_world_to_bins(Xw_cpu, R_wc, t_wc, K, W, H, bin_W, bin_H):
    """
    Xw_cpu: (M,3) float32 CPU
    반환: ui_bin, vi_bin, z_cam, valid  (모두 1D tensors)
    """
    # world -> cam
    # Xc = R * Xw + t
    Xc = (Xw_cpu @ R_wc.T) + t_wc  # (M,3)
    z = Xc[:, 2].clone()           # (M,)
    # 투영 (픽셀 좌표)
    x = Xc[:, 0] / (z.clamp_min(1e-8))
    y = Xc[:, 1] / (z.clamp_min(1e-8))
    u = K[0,0] * x + K[0,2]
    v = K[1,1] * y + K[1,2]

    # 이미지 내부 + z>0
    ui = u.long()
    vi = v.long()
    valid = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    if not valid.any():
        # 빈 텐서들 반환
        return (torch.empty(0, dtype=torch.long),)*2 + (torch.empty(0), torch.empty(0, dtype=torch.bool))

    # bin 인덱스 계산 (긴 변 기준 bin_max로 리스케일)
    su = bin_W / float(W)
    sv = bin_H / float(H)
    ui_bin = (u[valid] * su).floor().clamp(0, bin_W - 1).long()
    vi_bin = (v[valid] * sv).floor().clamp(0, bin_H - 1).long()
    return ui_bin, vi_bin, z[valid], valid[valid]  # valid[valid]는 전부 True지만 형식 통일

@torch.no_grad()

def _build_depth_maps_from_points(points_np, cams_meta, bin_max=64, chunk=200_000):
    """
    points_np: (M,3) numpy (월드 좌표, SfM 포인트들; 여기서는 filtered_xyz 사용)
    반환: 카메라마다 (bin_H, bin_W) 텐서 D0, 값은 '그 빈 방향의 포인트 최전방 depth(=z or -z)'.
    ※ 현재 코드는 OpenCV(+Z 전방) 기준. (네가 지금 쓰는 버전)
    """
    pts = torch.from_numpy(points_np).float()  # CPU (M,3)
    D0_list = []
    for cm in cams_meta:
        R_wc = cm["R_wc"]; t_wc = cm["t_wc"]; K = cm["K"]
        W, H = cm["W"], cm["H"]
        scale = bin_max / float(max(W, H))
        bin_W = max(1, int(round(W * scale)))
        bin_H = max(1, int(round(H * scale)))

        D0 = torch.full((bin_H, bin_W), float('inf'))
        D0_flat = D0.view(-1)

        for s in range(0, pts.shape[0], chunk):
            Xe = pts[s:s+chunk]
            # OpenCV(앞: z>0, 분모: z, depth=z) 투영
            Xc = (Xe @ R_wc.T) + t_wc
            z  = Xc[:, 2]
            x  = Xc[:, 0] / z.clamp_min(1e-8)
            y  = Xc[:, 1] / z.clamp_min(1e-8)
            u  = K[0,0] * x + K[0,2]
            v  = K[1,1] * y + K[1,2]

            ui = u.long(); vi = v.long()
            in_img  = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            valid   = (z > 0) & in_img
            if not valid.any():
                continue

            su = bin_W / float(W); sv = bin_H / float(H)
            ui_bin = (u[valid] * su).floor().clamp(0, bin_W-1).long()
            vi_bin = (v[valid] * sv).floor().clamp(0, bin_H-1).long()
            z_v    = z[valid]  # depth = z (작을수록 카메라와 가까움)

            pix_idx = vi_bin * bin_W + ui_bin
            D0_flat.scatter_reduce_(0, pix_idx, z_v, reduce='amin', include_self=True)

        D0_list.append(D0)
    return D0_list

@torch.no_grad()
@torch.no_grad()
def _flip_sign_via_visibility_with_margin(
    grid_xyz_cpu, sdf_cpu, cams_meta, D0_list,
    tau0, eps=1e-4, chunk=200_000, verbose=True
):
    """
    OpenCV(+Z) 기준:
    - D0는 'SfM 포인트의 최전방 z-깊이' (depth=z, z>0, amin)로 만들어져 있어야 함.
    - 음수 격자점을 투영해 zv를 구하고, zv < D0 - tau0 - eps 이면 양수로 플립.
    """
    # 0) 준비
    if grid_xyz_cpu.dtype != torch.float32:
        grid_xyz_cpu = grid_xyz_cpu.float()
    sdf_cpu = sdf_cpu.view(-1)
    neg_idx = (sdf_cpu < 0).nonzero(as_tuple=True)[0]
    if neg_idx.numel() == 0:
        return sdf_cpu.view(-1, 1)

    pts  = grid_xyz_cpu[neg_idx]               # (K,3) 음수 격자점들
    flip = torch.zeros(pts.shape[0], dtype=torch.bool)
    
    # --- NEW: '한 번이라도 보였는지' 추적하는 마스크 ---
    was_ever_visible = torch.zeros(pts.shape[0], dtype=torch.bool)
    # -----------------------------------------------

    # 디버그 카운터
    tot_valid, tot_ok = 0, 0

    for cam_i, (cm, D0) in enumerate(zip(cams_meta, D0_list)):
        R_wc = cm["R_wc"]; t_wc = cm["t_wc"]; K = cm["K"]
        W, H = cm["W"], cm["H"]
        bin_H, bin_W = D0.shape

        for s in range(0, pts.shape[0], chunk):
            Xe = pts[s:s+chunk]               # (chunk,3)

            # 1) world -> camera
            Xc = (Xe @ R_wc.T) + t_wc           # (chunk,3) CPU/float32
            z  = Xc[:, 2]
            in_front = (z > 0)                  # OpenCV(+Z)

            # 2) 투영
            z_safe = z.clamp_min(1e-8)
            u = K[0,0] * (Xc[:,0] / z_safe) + K[0,2]
            v = K[1,1] * (Xc[:,1] / z_safe) + K[1,2]

            ui = u.long(); vi = v.long()
            in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

            valid = in_front & in_img           # (chunk,)
            if not valid.any():
                continue

            # !!! 핵심: valid의 로컬 인덱스를 구해 정확히 매핑한다
            valid_idx = valid.nonzero(as_tuple=True)[0]     # (Nv,)

            # --- NEW: 'was_ever_visible' 마스크 업데이트 ---
            # (s + valid_idx)는 pts 텐서 기준의 전역 인덱스
            was_ever_visible[s + valid_idx] = True
            # -----------------------------------------------

            u_v = u[valid];  v_v = v[valid];  z_v = z[valid]  # (Nv,)

            # 3) 픽셀 -> bin
            su = bin_W / float(W);  sv = bin_H / float(H)
            ui_bin = (u_v * su).floor().clamp(0, bin_W-1).long()
            vi_bin = (v_v * sv).floor().clamp(0, bin_H-1).long()

            # 4) 그 bin의 표면(포인트) 최전방 깊이
            d0 = D0[vi_bin, ui_bin]                           # (Nv,)

            # 5) 플립 판정: zv < d0 - tau0 - eps
            # --- 기존 로직 (변경 없음) ---
            # (참고: 원본 코드에 tau0, eps가 빠져있어 그대로 둡니다)
            cond =  (z_v < d0) 
            if cond.any():
                # 청크 내 valid의 로컬 인덱스를 전역 neg_idx 기준 인덱스로 변환
                flip_idx_local = s + valid_idx[cond]        # (N_ok,)
                flip[flip_idx_local] = True
            # -----------------------------

            if verbose:
                tot_valid += int(valid.sum().item())
                tot_ok    += int(cond.sum().item())

        if verbose:
            print(f"[FLIP DEBUG] cam#{cam_i:02d} valid_acc={tot_valid} ok_acc={tot_ok}")

    # 6) 최종 반영
    sdf_new = sdf_cpu.clone()
    mask_full = torch.zeros_like(sdf_cpu, dtype=torch.bool)

    # --- MODIFIED: 최종 플립 조건 수정 ---
    # 1. 기존 'flip' (표면보다 앞에 보인 점)
    # 2. 'was_ever_visible'이 False인 점 (즉, 한 번도 보이지 않은 점)
    final_flip_condition = flip | (~was_ever_visible)
    mask_full[neg_idx] = final_flip_condition
    # --------------------------------------
    
    sdf_new[mask_full] = sdf_new[mask_full].abs()

    if verbose:
        n_total_neg = pts.shape[0]
        n_flip_visible_front = int(flip.sum().item())
        n_never_visible = int((~was_ever_visible).sum().item())
        n_flip_total = int(final_flip_condition.sum().item())
        
        print(f"[FLIP DEBUG] ----- Flip Summary -----")
        print(f"[FLIP DEBUG] Total negative verts: {n_total_neg}")
        print(f"[FLIP DEBUG] Flipped (visible & in front): {n_flip_visible_front} (Original logic)")
        print(f"[FLIP DEBUG] Flipped (never visible): {n_never_visible} (Added logic)")
        print(f"[FLIP DEBUG] TOTAL FLIPPED (to +): {n_flip_total} / {n_total_neg}")

    return sdf_new.view(-1, 1)