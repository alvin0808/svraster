import numpy as np
import torch

def select_K_keyframes_with_overlap(cur_cam, keyframes, K, N_samples=16, pixels=100):
    """
    Select K keyframes including cur_cam and recent keyframe, and K-2 with overlap.
    Works even when keyframes has fewer than K-2 entries.
    """
    

    device = cur_cam.c2w.device
    H, W = cur_cam.image_height, cur_cam.image_width
    fx = (W / 2) / np.tan(cur_cam.fovx / 2)
    fy = (H / 2) / np.tan(cur_cam.fovy / 2)
    cx = cur_cam.cx_p * W
    cy = cur_cam.cy_p * H
    #print(f"{H}x{W} | fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    #print(cur_cam.c2w)
    xs = torch.randint(0, W, (pixels,), device="cpu")
    ys = torch.randint(0, H, (pixels,), device="cpu")
    gt_depth = cur_cam.depth[0, ys, xs].unsqueeze(-1).to(device)
    gt_depth = gt_depth.reshape(-1, 1)
    gt_depth = gt_depth.repeat(1, N_samples)
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    near = gt_depth*0.8
    far = gt_depth+0.5
    #print(f"near={near.mean():.2f}, far={far.mean():.2f}, t_vals={t_vals.mean():.2f}")
    z_vals = near * (1.-t_vals) + far * (t_vals)
    i = xs.float()
    j = ys.float()

    dirs = torch.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        -torch.ones_like(i)
    ], dim=-1).to(device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    rays_d = torch.sum(dirs[..., None, :] * cur_cam.c2w[:3, :3], dim=-1)
    rays_o = cur_cam.c2w[:3, 3].expand_as(rays_d)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    vertices = pts.reshape(-1, 3).cpu().numpy()

    list_keyframe = []
    for keyframe_id, keyframe in enumerate(keyframes):
        w2c = keyframe.w2c.cpu().numpy()
        homo_vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
        cam_cord = (w2c @ homo_vertices.T).T[:, :3]
        cam_cord[:, 0] *= -1

        Kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        uv = (Kmat @ cam_cord.T).T
        z = uv[:, 2:] + 1e-5
        uv = uv[:, :2] / z

        edge = 20
        mask = (
            (uv[:, 0] > edge) & (uv[:, 0] < W - edge) &
            (uv[:, 1] > edge) & (uv[:, 1] < H - edge) &
            (cam_cord[:, 2] < 0)
        )
        percent_inside = mask.sum() / len(mask)
        if percent_inside > 0:
            list_keyframe.append({'id': keyframe_id, 'score': percent_inside})

    candidates = [entry['id'] for entry in list_keyframe]
    #print(candidates)
    np.random.shuffle(candidates)

    # 최근 keyframe 추가
    recent_keyframe = keyframes[-1] if len(keyframes) > 0 else None
    selected = []

    if recent_keyframe:
        selected.append(recent_keyframe)

    # 나머지 선택
    for idx in candidates:
        if len(selected) >= K - 1:
            break
        if keyframes[idx] != recent_keyframe:
            selected.append(keyframes[idx])

    # cur_cam까지 포함해서 총 K개
    selected.append(cur_cam)


    return selected[:K]
