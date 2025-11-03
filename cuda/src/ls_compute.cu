#include "ls_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace LS_COMPUTE {
// 1. CUDA kernel


__global__ void grid_laplacian_kernel(
    const float* __restrict__ grid_pts,       // [M, 1]
    const int64_t* __restrict__ vox_key,     // [N, 8]
    const int32_t* __restrict__ grid_keys,    // [G]
    const int32_t* __restrict__ grid2voxel,   // [G]
    const int32_t* __restrict__  active_list, // [M]
    const int32_t grid_res,                       // grid resolution
    const bool* __restrict__ grid_mask,       // [grid_res * grid_res * grid_res]
    const float* __restrict__ voxel_coords,   // [N, 3]
    const float* __restrict__ voxel_sizes,    // [N]
    const int M,                              // number of grid points
    const int A,                              // number of active grid points
    const int num_voxels,                     // number of voxels
    const int grid_pts_size,                  // size of grid points
    const float weight,
    const float vox_size_inv,
    float* __restrict__ grid_pts_grad         // [M, 1]
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= A) return;
    //if(voxel_sizes[grid2voxel[tid]] > 2.0f) return;
    int gk = grid_keys[active_list[tid]];
    int x = gk % grid_res;
    int y = (gk / grid_res) % grid_res;
    int z = gk / (grid_res * grid_res);
    if (!grid_mask[gk] || !check_valid_neighbors(x, y, z, grid_res, grid_mask)) return;

    float w[7][8];
    int vox_id[7];
    int seed = (int)clock64();
    float3 randv=get_rand_vec(tid, seed);
    float neighbor_sdfs[6];
    int dx[6] = {+1, -1, 0, 0, 0, 0};
    int dy[6] = {0, 0, +1, -1, 0, 0};
    int dz[6] = {0, 0, 0, 0, +1, -1};

    float center_sdf = interpolate_sdf_and_grad(M,
        x, y, z,
        grid_keys, grid2voxel,
        voxel_coords, voxel_sizes,
        vox_key, grid_pts, grid_res,
        num_voxels, grid_pts_size,
        w[0], vox_id[0],randv); // center weight and id (dummy reuse)
    
    for (int i = 0; i < 6; ++i) {
        neighbor_sdfs[i] = interpolate_sdf_and_grad(M,
            x + dx[i], y + dy[i], z + dz[i],
            grid_keys, grid2voxel,
            voxel_coords, voxel_sizes,
            vox_key, grid_pts, grid_res,
            num_voxels, grid_pts_size,
            w[i+1], vox_id[i+1],randv);
    }

    // for (int i = 0; i < 6; ++i) sdf_avg += neighbor_sdfs[i];
    // sdf_avg /= 6.0f;
    float lap_residual_x = neighbor_sdfs[0] + neighbor_sdfs[1] - 2.0f * center_sdf;
    float lap_residual_y = neighbor_sdfs[2] + neighbor_sdfs[3] - 2.0f * center_sdf;
    float lap_residual_z = neighbor_sdfs[4] + neighbor_sdfs[5] - 2.0f * center_sdf;

    // size
    float lap_residual = sqrtf(lap_residual_x * lap_residual_x +
                          lap_residual_y * lap_residual_y +
                          lap_residual_z * lap_residual_z + 1e-10f); // avoid division by zero
    float additional_weight =(512.0f / grid_res);
    float size =  weight* 0.5*vox_size_inv*vox_size_inv ; // 512/grid_res -> to balance different resolution
    float dL_dx = 0.5 / lap_residual * 2* lap_residual_x * size; // gradient w.r.t. f(x+1) and f(x-1)
    float dL_dy = 0.5 / lap_residual * 2* lap_residual_y * size;
    float dL_dz = 0.5 / lap_residual * 2* lap_residual_z * size;
    float dL_dcenter = -2.0f * (dL_dx + dL_dy + dL_dz);

 


    // float lap_residual = center_sdf - sdf_avg; //  f(x) - mean(neighbors)
    // float dL_df = 2.0f * lap_residual * weight; // gradient w.r.t. f(x_i)
    /*
    accumulate_grad(vox_id[0], num_voxels, vox_key, w[0], dL_dcenter/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]], grid_pts_grad);
    accumulate_grad(vox_id[1], num_voxels, vox_key, w[1], dL_dx/voxel_sizes[vox_id[1]]/voxel_sizes[vox_id[1]]/voxel_sizes[vox_id[0]], grid_pts_grad);
    accumulate_grad(vox_id[2], num_voxels, vox_key, w[2], dL_dx/voxel_sizes[vox_id[2]]/voxel_sizes[vox_id[2]]/voxel_sizes[vox_id[0]], grid_pts_grad);
    accumulate_grad(vox_id[3], num_voxels, vox_key, w[3], dL_dy/voxel_sizes[vox_id[3]]/voxel_sizes[vox_id[3]]/voxel_sizes[vox_id[0]], grid_pts_grad);
    accumulate_grad(vox_id[4], num_voxels, vox_key, w[4], dL_dy/voxel_sizes[vox_id[4]]/voxel_sizes[vox_id[4]]/voxel_sizes[vox_id[0]], grid_pts_grad);
    accumulate_grad(vox_id[5], num_voxels, vox_key, w[5], dL_dz/voxel_sizes[vox_id[5]]/voxel_sizes[vox_id[5]]/voxel_sizes[vox_id[0]], grid_pts_grad);
    accumulate_grad(vox_id[6], num_voxels, vox_key, w[6], dL_dz/voxel_sizes[vox_id[6]]/voxel_sizes[vox_id[6]]/voxel_sizes[vox_id[0]], grid_pts_grad);
    */
    
    accumulate_grad(vox_id[0], num_voxels, vox_key, w[0], dL_dcenter, grid_pts_grad);
    accumulate_grad(vox_id[1], num_voxels, vox_key, w[1], dL_dx, grid_pts_grad);
    accumulate_grad(vox_id[2], num_voxels, vox_key, w[2], dL_dx, grid_pts_grad);
    accumulate_grad(vox_id[3], num_voxels, vox_key, w[3], dL_dy, grid_pts_grad);
    accumulate_grad(vox_id[4], num_voxels, vox_key, w[4], dL_dy, grid_pts_grad);
    accumulate_grad(vox_id[5], num_voxels, vox_key, w[5], dL_dz, grid_pts_grad);
    accumulate_grad(vox_id[6], num_voxels, vox_key, w[6], dL_dz, grid_pts_grad);
    
        
}

// 2. C++ 
void laplacian_smoothness_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const torch::Tensor& grid_voxel_coord,
    const torch::Tensor& grid_voxel_size,
    const int32_t grid_res,
    const torch::Tensor& grid_mask,
    const torch::Tensor& grid_keys,
    const torch::Tensor& grid2voxel,
    const torch::Tensor& active_list,
    const float weight,
    const float vox_size_inv,
    const bool no_tv_s,
    const bool tv_sparse,
    const torch::Tensor& grid_pts_grad
) {
    // Launch CUDA kernel
    const int M = grid_keys.size(0);
    const int A = active_list.size(0);
    const int threads = 256;
    const int blocks = (A + threads - 1) / threads;
    int num_voxels = vox_key.size(0);
    int grid_pts_size = grid_pts.size(0);
    grid_laplacian_kernel<<<blocks, threads>>>(
        grid_pts.contiguous().data_ptr<float>(),
        vox_key.contiguous().data_ptr<int64_t>(),
        grid_keys.contiguous().data_ptr<int32_t>(),
        grid2voxel.contiguous().data_ptr<int32_t>(),
        active_list.contiguous().data_ptr<int32_t>(),
        grid_res,
        grid_mask.contiguous().data_ptr<bool>(),
        grid_voxel_coord.contiguous().data_ptr<float>(),
        grid_voxel_size.contiguous().data_ptr<float>(),
        M,
        A,
        num_voxels,
        grid_pts_size,
        weight,
        vox_size_inv,
        grid_pts_grad.contiguous().data_ptr<float>()
    );
}

}
