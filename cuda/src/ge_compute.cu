#include "ge_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace GE_COMPUTE {
// 1. CUDA 커널


__global__ void grid_eikonal_kernel(
    const float* __restrict__ grid_pts,       // [M, 1]
    const int64_t* __restrict__ vox_key,     // [N, 8]
    const int32_t* __restrict__ grid_keys,    // [G]
    const int32_t* __restrict__ grid2voxel,   // [G]
    const int32_t grid_res,                       // grid resolution
    const bool* __restrict__ grid_mask,       // [grid_res³]
    const float* __restrict__ voxel_coords,   // [N, 3]
    const float* __restrict__ voxel_sizes,    // [N]
    const int M,                              // number of grid points
    const int num_voxels,                     // number of voxels
    const int grid_pts_size,                  // size of grid points
    const float weight,
    const float vox_size_inv,
    float* __restrict__ grid_pts_grad         // [M, 1]
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    int gk = grid_keys[tid];
    int x = gk % grid_res;
    int y = (gk / grid_res) % grid_res;
    int z = gk / (grid_res * grid_res);
    if (!grid_mask[gk] || !check_valid_neighbors(x, y, z, grid_res, grid_mask)) return;

    float w[6][8];
    int vox_id[6];

    int dx[6] = {+1, -1, 0, 0, 0, 0};
    int dy[6] = {0, 0, +1, -1, 0, 0};
    int dz[6] = {0, 0, 0, 0, +1, -1};

    float sdfs[6];
    for (int i = 0; i < 6; ++i) {
        sdfs[i] = interpolate_sdf_and_grad(M,
            x + dx[i], y + dy[i], z + dz[i],
            grid_keys, grid2voxel,
            voxel_coords, voxel_sizes,
            vox_key, grid_pts, grid_res,
            num_voxels, grid_pts_size,
            w[i], vox_id[i]);
    }
    //if(grid_res == 128) printf("ok");

    float dx_val = (sdfs[0] - sdfs[1]) * 0.5f;
    float dy_val = (sdfs[2] - sdfs[3]) * 0.5f;
    float dz_val = (sdfs[4] - sdfs[5]) * 0.5f;

    float grad_norm = sqrtf(dx_val*dx_val + dy_val*dy_val + dz_val*dz_val + 1e-10f);
    float grad_world = grad_norm * 0.5*vox_size_inv;
    float dL_dg = 2.0f * (grad_world - 1.0f) * 0.5*vox_size_inv * weight;

    float dL_dx = dL_dg * dx_val / grad_norm * 0.5f;
    float dL_dy = dL_dg * dy_val / grad_norm * 0.5f;
    float dL_dz = dL_dg * dz_val / grad_norm * 0.5f;
    /*
    accumulate_grad(vox_id[0], num_voxels, vox_key, w[0], dL_dx, grid_pts_grad);
    accumulate_grad(vox_id[1], num_voxels,vox_key, w[1], -dL_dx, grid_pts_grad);
    accumulate_grad(vox_id[2], num_voxels,vox_key, w[2], dL_dy, grid_pts_grad);
    accumulate_grad(vox_id[3], num_voxels,vox_key, w[3], -dL_dy, grid_pts_grad);
    accumulate_grad(vox_id[4], num_voxels,vox_key, w[4], dL_dz, grid_pts_grad);
    accumulate_grad(vox_id[5], num_voxels,vox_key, w[5], -dL_dz, grid_pts_grad);*/

}

// 2. C++ 인터페이스
void grid_eikonal_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const torch::Tensor& grid_voxel_coord,
    const torch::Tensor& grid_voxel_size,
    const int32_t grid_res,
    const torch::Tensor& grid_mask,
    const torch::Tensor& grid_keys,
    const torch::Tensor& grid2voxel,
    const float weight,
    const float vox_size_inv,
    const bool no_tv_s,
    const bool tv_sparse,
    const torch::Tensor& grid_pts_grad
) {
    // Launch CUDA kernel
    const int M = grid_keys.size(0);
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;
    int num_voxels = vox_key.size(0);
    int grid_pts_size = grid_pts.size(0);
    grid_eikonal_kernel<<<blocks, threads>>>(
        grid_pts.contiguous().data_ptr<float>(),
        vox_key.contiguous().data_ptr<int64_t>(),
        grid_keys.contiguous().data_ptr<int32_t>(),
        grid2voxel.contiguous().data_ptr<int32_t>(),
        grid_res,
        grid_mask.contiguous().data_ptr<bool>(),
        grid_voxel_coord.contiguous().data_ptr<float>(),
        grid_voxel_size.contiguous().data_ptr<float>(),
        M,
        num_voxels,
        grid_pts_size,
        weight,
        vox_size_inv,
        grid_pts_grad.contiguous().data_ptr<float>()
    );
}

}
