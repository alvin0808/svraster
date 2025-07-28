/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "vg_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace VG_COMPUTE {


__global__ void voxel_gradient_kernel(
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
    if(voxel_sizes[grid2voxel[tid]] > 2.0f) return;
    /*
    int gk = grid_keys[tid];
    int x = gk % grid_res;
    int y = (gk / grid_res) % grid_res;
    int z = gk / (grid_res * grid_res);
    if (!grid_mask[gk] || !check_valid_neighbors(x, y, z, grid_res, grid_mask)) return;

    int vox_id = grid2voxel[tid];
    int64_t grid_pts_idx[8];
    float sdfs[8];
    float3 grad = make_float3(0.0f, 0.0f, 0.0f);
    float w[8][3];
    int seed = (int)clock64();
    float3 randv=get_rand_vec(tid, seed);
    
    tri_interp_grad_weight(randv, w);
    for(int i=0;i<8;i++){
        grid_pts_idx[i] = vox_key[vox_id * 8 + i];
        sdfs[i]=grid_pts[grid_pts_idx[i]];
        grad.x += w[i][0]*sdfs[i];
        grad.y += w[i][1]*sdfs[i];
        grad.z += w[i][2]*sdfs[i];
    }
    float adj_sdfs[6];
    int adj_vox_id[6];
    float adj_w[6][8];
    int dx[6] = {+1, -1, 0, 0, 0, 0};
    int dy[6] = {0, 0, +1, -1, 0, 0};
    int dz[6] = {0, 0, 0, 0, +1, -1};
    randv=get_rand_vec(tid, seed*2);
    for (int i = 0; i < 6; ++i) {
        adj_sdfs[i] = interpolate_sdf_and_grad(M,
            x + dx[i], y + dy[i], z + dz[i],
            grid_keys, grid2voxel,
            voxel_coords, voxel_sizes,
            vox_key, grid_pts, grid_res,
            num_voxels, grid_pts_size,
            adj_w[i], adj_vox_id[i],randv);
    }
    //if(grid_res == 128) printf("ok");

    float dx_val = (adj_sdfs[0] - adj_sdfs[1]) *0.5f;
    float dy_val = (adj_sdfs[2] - adj_sdfs[3]) *0.5f;
    float dz_val = (adj_sdfs[4] - adj_sdfs[5]) *0.5f;
    float dx_diff = grad.x-dx_val;
    float dy_diff = grad.y-dy_val;
    float dz_diff = grad.z-dz_val;
    float grad_coeff = 2.0f * weight * vox_size_inv;
    float dL_dgx = grad_coeff * dx_diff;
    float dL_dgy = grad_coeff * dy_diff;
    float dL_dgz = grad_coeff * dz_diff;
    for (int i = 0; i < 8; ++i) {
        float g = dL_dgx * w[i][0] + dL_dgy * w[i][1] + dL_dgz * w[i][2];
        atomicAdd(&grid_pts_grad[grid_pts_idx[i]], g);
    }
    accumulate_grad(adj_vox_id[0], num_voxels, vox_key, adj_w[0], -dL_dgx, grid_pts_grad);
    accumulate_grad(adj_vox_id[1], num_voxels,vox_key, adj_w[1], dL_dgx, grid_pts_grad);
    accumulate_grad(adj_vox_id[2], num_voxels,vox_key, adj_w[2], -dL_dgy, grid_pts_grad);
    accumulate_grad(adj_vox_id[3], num_voxels,vox_key, adj_w[3], dL_dgy, grid_pts_grad);
    accumulate_grad(adj_vox_id[4], num_voxels,vox_key, adj_w[4], -dL_dgz, grid_pts_grad);
    accumulate_grad(adj_vox_id[5], num_voxels,vox_key, adj_w[5], dL_dgz, grid_pts_grad);
    */
    
    int gk = grid_keys[tid];
    int x = gk % grid_res;
    int y = (gk / grid_res) % grid_res;
    int z = gk / (grid_res * grid_res);
    if (!grid_mask[gk] ) return;
    if ( x >= grid_res - 1 || y >= grid_res - 1 || z >= grid_res - 1)
        return;
    if(!(grid_mask[flatten_grid_key(x+1, y, z, grid_res)] && grid_mask[flatten_grid_key(x, y+1, z, grid_res)] &&grid_mask[flatten_grid_key(x, y, z+1, grid_res)]) )
        return;
    float w[8][3];

    int dx[3] = {+1, 0, 0};
    int dy[3] = {0, +1, 0};
    int dz[3] = {0, 0, +1};
    int seed = (int)clock64();
    float3 randv=get_rand_vec(tid, seed);
    int vox_id[4];
    vox_id[0] = grid2voxel[tid];
    for(int i=0;i<3;i++){
        int gk = flatten_grid_key(x+dx[i], y+dy[i], z+dz[i], grid_res);
        int idx = binary_search(grid_keys, M , gk);
        vox_id[i+1] = grid2voxel[idx];
    }
    tri_interp_grad_weight(randv, w);
    float grad[4][3];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        grad[i][0] = grad[i][1] = grad[i][2] = 0.0f;
    for(int i=0;i<4;i++){
        int64_t grid_pts_idx[8];
        float sdfs[8];
        for(int j=0;j<8;j++){
            grid_pts_idx[j] = vox_key[vox_id[i]*8+j];
            sdfs[j]=grid_pts[grid_pts_idx[j]];
            grad[i][0] += w[j][0]*sdfs[j];
            grad[i][1] += w[j][1]*sdfs[j];
            grad[i][2] += w[j][2]*sdfs[j];
        }
    } 
    //float grad_diff[3];
    //grad_diff[0] = (grad[0][0] - grad[1][0])*(grad[0][0] - grad[1][0]) + (grad[0][1] - grad[1][1])*(grad[0][1]- grad[1][1]) + (grad[0][2] - grad[1][2])*(grad[0][2] - grad[1][2]);  
    //grad_diff[1] = (grad[0][0] - grad[2][0])*(grad[0][0] - grad[2][0]) + (grad[0][1] - grad[2][1])*(grad[0][1]- grad[2][1]) + (grad[0][2] - grad[2][2])*(grad[0][2] - grad[2][2]);
    //grad_diff[2] = (grad[0][0] - grad[3][0])*(grad[0][0] - grad[3][0]) + (grad[0][1] - grad[3][1])*(grad[0][1]- grad[3][1]) + (grad[0][2] - grad[3][2])*(grad[0][2] - grad[3][2]);
    for (int d = 0; d < 3; ++d) {
        float diff = grad[0][d] - grad[1][d];
        float coeff = 2.f * diff * weight *vox_size_inv;
        for (int j = 0; j < 8; ++j) {
            atomicAdd(&grid_pts_grad[vox_key[vox_id[0] * 8 + j]], coeff * w[j][d]/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]]);
            atomicAdd(&grid_pts_grad[vox_key[vox_id[1] * 8 + j]], -coeff * w[j][d]/voxel_sizes[vox_id[1]]/voxel_sizes[vox_id[1]]/voxel_sizes[vox_id[1]]);
        }

        diff = grad[0][d] - grad[2][d];
        coeff = 2.f * diff * weight*vox_size_inv;
        for (int j = 0; j < 8; ++j) {
            atomicAdd(&grid_pts_grad[vox_key[vox_id[0] * 8 + j]], coeff * w[j][d]/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]]);
            atomicAdd(&grid_pts_grad[vox_key[vox_id[2] * 8 + j]], -coeff * w[j][d]/voxel_sizes[vox_id[2]]/voxel_sizes[vox_id[2]]/voxel_sizes[vox_id[2]]);
        }

        diff = grad[0][d] - grad[3][d];
        coeff = 2.f * diff * weight*vox_size_inv;
        for (int j = 0; j < 8; ++j) {
            atomicAdd(&grid_pts_grad[vox_key[vox_id[0] * 8 + j]], coeff * w[j][d]/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]]/voxel_sizes[vox_id[0]]);
            atomicAdd(&grid_pts_grad[vox_key[vox_id[3] * 8 + j]], -coeff * w[j][d]/voxel_sizes[vox_id[3]]/voxel_sizes[vox_id[3]]/voxel_sizes[vox_id[3]]);
        }
    }

    

}

// 2. C++ 인터페이스
void voxel_gradient_bw(
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
    voxel_gradient_kernel<<<blocks, threads>>>(
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
