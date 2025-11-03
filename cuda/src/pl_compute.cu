#include "pl_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace PL_COMPUTE {
// 1. CUDA 커널


__global__ void points_loss_kernel(
    const float* __restrict__ points_in_grid, // [M, 3]
    const float* __restrict__ grid_pts,       // 
    const int64_t* __restrict__ vox_key,     // [N, 8]
    const int32_t* __restrict__ grid_keys,    // [G]
    const int32_t* __restrict__ grid2voxel,   // [G]
    const int32_t grid_res,                       // grid resolution
    const bool* __restrict__ grid_mask,       // [grid_res³]
    const float* __restrict__ voxel_coords,   // [N, 3]
    const float* __restrict__ voxel_sizes,    // [N]
    const int M,                              // number of grid points
    const int num_voxels,                     // number of voxels
    const int grid_keys_size,                  // size of grid points
    const float weight,
    const float vox_size_inv,
    float* __restrict__ grid_pts_grad         // [M, 1]
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;
    float3 pg = make_float3(points_in_grid[tid * 3 + 0],
                          points_in_grid[tid * 3 + 1],
                          points_in_grid[tid * 3 + 2]);
    int ix = static_cast<int>(pg.x);
    int iy = static_cast<int>(pg.y);
    int iz = static_cast<int>(pg.z);
    int pg_id = ix+iy*grid_res+iz*grid_res*grid_res;
    if(ix<0 || ix>=grid_res || iy<0 || iy>=grid_res || iz<0 || iz>=grid_res) return;
    if(grid_mask[pg_id]==false) return;
    int idx = binary_search(grid_keys,grid_keys_size, pg_id);
    int vox_id = grid2voxel[idx];
    float voxel_size = voxel_sizes[vox_id];
    float3 voxel_coord = make_float3(voxel_coords[vox_id * 3 + 0],
                                      voxel_coords[vox_id * 3 + 1],
                                      voxel_coords[vox_id * 3 + 2]);
    float3 local_pos = make_float3((pg.x - voxel_coord.x)/voxel_size,
                                   (pg.y - voxel_coord.y)/voxel_size,
                                   (pg.z - voxel_coord.z)/voxel_size);
    float weights_out[8];
    tri_interp_weight(local_pos, weights_out);
    float sdf=0.0f;
    for(int i=0;i<8;i++){
        int64_t gk = vox_key[vox_id * 8 + i];
        sdf += weights_out[i]*grid_pts[gk];
    }
    //calculated the loss (sdf-0)^2=L
    float loss = (sdf - 0.0f) * (sdf - 0.0f);
    float dL_dg = 2.0f * (sdf - 0.0f) * weight;
    accumulate_grad(vox_id, num_voxels, vox_key, weights_out, dL_dg, grid_pts_grad);

}

// 2. C++ 
void points_loss_bw(
    const torch::Tensor& points_in_grid,
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
    const int M = points_in_grid.size(0);
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;
    int num_voxels = vox_key.size(0);
    int grid_keys_size = grid_keys.size(0);
    points_loss_kernel<<<blocks, threads>>>(
        points_in_grid.contiguous().data_ptr<float>(),
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
        grid_keys_size,
        weight,
        vox_size_inv,
        grid_pts_grad.contiguous().data_ptr<float>()
    );
}

}
