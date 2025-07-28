/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "tv_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace TV_COMPUTE {

template <bool no_tv_s, bool tv_sparse>
__global__ void total_variation_bw_cuda(
    const int N, const int C, const int NC,
    const float* __restrict__ grid_pts,
    const int64_t* __restrict__ vox_key,
    const float weight,
    const float* __restrict__ vox_size_inv,
    float* __restrict__ grid_pts_grad)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= NC)
        return;
    const int iN = idx / C;
    const int iC = idx % C;

    // Load from global memory.
    int i_book[8];
    #pragma unroll
    for (int i=0, k=iN*8; i<8; ++i, ++k)
        i_book[i] = vox_key[k];

    if (tv_sparse)
    {
        bool valid = false;
        for (int i=0; i<8; ++i)
            valid |= (grid_pts_grad[i_book[i] * C + iC] != 0.f);
        if (!valid)
            return;
    }

    float vlst[8];
    #pragma unroll
    for (int i=0; i<8; ++i)
        vlst[i] = grid_pts[i_book[i] * C + iC];

    
    //bool has_surface = false;
    //float sign0 = (vlst[0] > 0.f);
    //for (int i = 1; i < 8; ++i)
    //    has_surface |= ((vlst[i] > 0.f) != sign0);

    float w = weight;
    //if (has_surface)
    //    w *= 3.0f;

    
    
    if (!no_tv_s)
        w *= 0.01f * vox_size_inv[iN];
    
    // Compute gradient wrt total variation loss
    
    int glst[8] = {0};
    #pragma unroll
    for (int i=0; i<8; ++i)
    {
        glst[i] += (vlst[i] > vlst[i^0b001]) * 2 - 1;
        glst[i] += (vlst[i] > vlst[i^0b010]) * 2 - 1;
        glst[i] += (vlst[i] > vlst[i^0b100]) * 2 - 1;
    }

    float dtv_dgrid_pts[8];
    #pragma unroll
    for (int i=0; i<8; ++i)
        dtv_dgrid_pts[i] = w * ((float)glst[i]);
    
    /*
    float edge_diff[12];
    edge_diff[0] = vlst[0] - vlst[1]; // x-edge
    edge_diff[1] = vlst[2] - vlst[3]; // x-edge
    edge_diff[2] = vlst[4] - vlst[5]; // x-edge
    edge_diff[3] = vlst[6] - vlst[7]; // x-edge

    edge_diff[4] = vlst[0] - vlst[2]; // y-edge
    edge_diff[5] = vlst[1] - vlst[3]; // y-edge
    edge_diff[6] = vlst[4] - vlst[6]; // y-edge
    edge_diff[7] = vlst[5] - vlst[7]; // y-edge

    edge_diff[8]  = vlst[0] - vlst[4]; // z-edge
    edge_diff[9]  = vlst[1] - vlst[5]; // z-edge
    edge_diff[10] = vlst[2] - vlst[6]; // z-edge
    edge_diff[11] = vlst[3] - vlst[7]; // z-edge
    float mean_x = 0.25f * (edge_diff[0] + edge_diff[1] + edge_diff[2] + edge_diff[3]);
    float mean_y = 0.25f * (edge_diff[4] + edge_diff[5] + edge_diff[6] + edge_diff[7]);
    float mean_z = 0.25f * (edge_diff[8] + edge_diff[9] + edge_diff[10] + edge_diff[11]);
    float grad_parallel[8] = {0.f};  // parallel loss에 대한 gradient 초기화

    // x 방향 모서리
    grad_parallel[0] += 2.0f * (edge_diff[0] - mean_x);
    grad_parallel[1] -= 2.0f * (edge_diff[0] - mean_x);
    grad_parallel[2] += 2.0f * (edge_diff[1] - mean_x);
    grad_parallel[3] -= 2.0f * (edge_diff[1] - mean_x);
    grad_parallel[4] += 2.0f * (edge_diff[2] - mean_x);
    grad_parallel[5] -= 2.0f * (edge_diff[2] - mean_x);
    grad_parallel[6] += 2.0f * (edge_diff[3] - mean_x);
    grad_parallel[7] -= 2.0f * (edge_diff[3] - mean_x);

    // y 방향 모서리
    grad_parallel[0] += 2.0f * (edge_diff[4] - mean_y);
    grad_parallel[2] -= 2.0f * (edge_diff[4] - mean_y);
    grad_parallel[1] += 2.0f * (edge_diff[5] - mean_y);
    grad_parallel[3] -= 2.0f * (edge_diff[5] - mean_y);
    grad_parallel[4] += 2.0f * (edge_diff[6] - mean_y);
    grad_parallel[6] -= 2.0f * (edge_diff[6] - mean_y);
    grad_parallel[5] += 2.0f * (edge_diff[7] - mean_y);
    grad_parallel[7] -= 2.0f * (edge_diff[7] - mean_y);

    // z 방향 모서리
    grad_parallel[0] += 2.0f * (edge_diff[8] - mean_z);
    grad_parallel[4] -= 2.0f * (edge_diff[8] - mean_z);
    grad_parallel[1] += 2.0f * (edge_diff[9] - mean_z);
    grad_parallel[5] -= 2.0f * (edge_diff[9] - mean_z);
    grad_parallel[2] += 2.0f * (edge_diff[10] - mean_z);
    grad_parallel[6] -= 2.0f * (edge_diff[10] - mean_z);
    grad_parallel[3] += 2.0f * (edge_diff[11] - mean_z);
    grad_parallel[7] -= 2.0f * (edge_diff[11] - mean_z);
    float dtv_dgrid_pts[8] = {0.f};
    for (int i = 0; i < 8; ++i) {
        dtv_dgrid_pts[i] += w * grad_parallel[i];
    }*/
    
  /*
    float dtv_dgrid_pts[8] = {0.f};

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float dx = 0.5f * (vlst[i ^ 0b001] + vlst[i]) - vlst[i];  // x 방향 차이
        float dy = 0.5f * (vlst[i ^ 0b010] + vlst[i]) - vlst[i];  // y 방향 차이
        float dz = 0.5f * (vlst[i ^ 0b100] + vlst[i]) - vlst[i];  // z 방향 차이
        dtv_dgrid_pts[i] = w * (dx + dy + dz);
    }*/
    
    // Write back
    #pragma unroll
    for (int i=0; i<8; ++i)
        atomicAdd(grid_pts_grad + i_book[i] * C + iC, dtv_dgrid_pts[i]);
}


// Python interface to directly write the gradient of tv loss.
void total_variation_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const float weight,
    const torch::Tensor& vox_size_inv,
    const bool no_tv_s,
    const bool tv_sparse,
    const torch::Tensor& grid_pts_grad)
{
    const int N = vox_key.size(0);
    const int C = grid_pts.size(1);
    const int NC = N * C;

    auto tv_kernel =
        (no_tv_s & tv_sparse) ? total_variation_bw_cuda<true, true>   :
        (no_tv_s)             ? total_variation_bw_cuda<true, false>  :
        (tv_sparse)           ? total_variation_bw_cuda<false, true>  :
                                total_variation_bw_cuda<false, false> ;

    if (N > 0)
        tv_kernel <<<(NC + 255) / 256, 256>>> (
            N, C, NC,
            grid_pts.contiguous().data_ptr<float>(),
            vox_key.contiguous().data_ptr<int64_t>(),
            weight,
            vox_size_inv.contiguous().data_ptr<float>(),
            grid_pts_grad.contiguous().data_ptr<float>());
}

}
