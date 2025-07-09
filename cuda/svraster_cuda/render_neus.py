# render_neus.py
# ------------------------------------------------------------
# NeuS-style volume renderer in PyTorch for SVRaster
# Called when density_mode == 'sdf'
# ------------------------------------------------------------

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


# ------------------------
# Utility: Ray Generator
# ------------------------
def build_rays(H: int, W: int, rast, device) -> Tuple[torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    dirs = torch.stack([
        (i - rast.cx) / rast.tanfovx,
        (j - rast.cy) / rast.tanfovy,
        torch.ones_like(i)
    ], -1)  # [H, W, 3]
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    R = rast.c2w_matrix[:3, :3]  # [3,3]
    rays_d = (dirs @ R.T).reshape(-1, 3)  # [H*W, 3]
    rays_o = rast.c2w_matrix[:3, 3].expand_as(rays_d)  # [H*W, 3]
    return rays_o, rays_d


# ------------------------
# Main Entry Point
# ------------------------
'''
raster_settings = svraster_cuda.renderer.RasterSettings(
            color_mode=color_mode,
            vox_geo_mode=self.vox_geo_mode,
            density_mode=self.density_mode,
            image_width=w,
            image_height=h,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            cx=camera.cx * w_ss,
            cy=camera.cy * h_ss,
            w2c_matrix=camera.w2c,
            c2w_matrix=camera.c2w,
            background=self.bg_color,
            cam_mode=camera.cam_mode,
            near=camera.near,
            need_depth=output_depth,
            need_normal=output_normal,
            track_max_w=track_max_w,
            **other_opt)
'''
def render_neus(
    raster_settings,  #
    geomBuffer,       #
    octree_paths,     # Tensor [N], morton code of each voxel
    vox_centers,      # Tensor [N, 3], center position of each voxel
    vox_lengths,      # Tensor [N, 1], edge length of each voxel
    geos,             # Tensor [N, 8], SDF value at each voxel's 8 corners
    rgbs,             # Tensor [N, 3] or [N, 8, 3], RGB or SH color
    subdiv_p,         # Tensor [N, 1], dummy param for gradient tracking
    *,
    n_samples: int = 64, #?
    log_s: float = 100.0,#?
):
    H, W = raster_settings.image_height, raster_settings.image_width
    device = vox_centers.device

    # Step 1. Ray 생성
    ray_o, ray_d = build_rays(H, W, raster_settings, device)
    t_min = torch.full((H * W,), raster_settings.near, device=device)
    t_max = torch.full_like(t_min, 1.5)  # 필요 시 계산식으로 수정

    # Step 2. Voxel 기반 sample 정보 구성 (미구현, 샘플용 dummy)
    vox_info = build_voxinfo_for_neus(
        geomBuffer, octree_paths, vox_centers, vox_lengths,
        geos, rgbs, ray_o, ray_d, t_min, t_max, n_samples
    )

    # Step 3. Core NeuS-style rendering
    color, depth, normal, T_last = _core_render_neus(
        ray_o, ray_d, t_min, t_max, vox_info,
        n_samples=n_samples, log_s=log_s
    )

    # Step 4. reshape to (C, H, W)
    color = color.view(H, W, 3).permute(2, 0, 1).contiguous()
    depth = depth.view(H, W).contiguous()
    T_last = T_last.view(H, W).contiguous()

    return color, depth, normal, T_last, None, torch.empty(0, device=device)


# ------------------------
# Placeholder (to implement)
# ------------------------
def _core_render_neus(ray_o, ray_d, t_min, t_max, vox_info, n_samples=64, log_s=100.0):
    raise NotImplementedError("NeuS core renderer not implemented yet.")

def build_voxinfo_for_neus(
    geomBuffer, octree_paths, vox_centers, vox_lengths,
    geos, rgbs, ray_o, ray_d, t_min, t_max, n_samples
):
    raise NotImplementedError("Voxel-sample mapping not implemented.")