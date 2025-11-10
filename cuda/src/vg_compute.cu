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
    const float* __restrict__ vox_size_inv,    // [N]
    const int32_t* __restrict__  active_list, // [A]
    const int A,                              // number of active grid points
    const float weight,
    float* __restrict__ grid_pts_grad         // [M, 1]
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= A) return;

    const int v = active_list[tid];   // voxel id
    const float inv4h = 0.25f * vox_size_inv[v]; // 1/(4h)

    // Corner order is: [000,001,010,011,100,101,110,111]
    // Precomputed sign tables for (2*bx-1), (2*by-1), (2*bz-1) in that order:
    const float sx[8] = {-1.f,-1.f,-1.f,-1.f, +1.f,+1.f,+1.f,+1.f};
    const float sy[8] = {-1.f,-1.f, +1.f, +1.f, -1.f,-1.f, +1.f, +1.f};
    const float sz[8] = {-1.f, +1.f,-1.f, +1.f, -1.f, +1.f,-1.f, +1.f};

    // 1) Load SDF at the 8 corners
    float S[8];
    #pragma unroll
    for (int i=0;i<8;++i){
        const int64_t gidx = vox_key[v*8 + i];
        S[i] = grid_pts[gidx];
    }

    // 2) Gradient at voxel center via 4-edge-avg central differences
    float gx = 0.f, gy = 0.f, gz = 0.f;
    #pragma unroll
    for (int i=0;i<8;++i){
        gx += sx[i] * S[i];
        gy += sy[i] * S[i];
        gz += sz[i] * S[i];
    }
    gx *= inv4h; gy *= inv4h; gz *= inv4h;

    // 3) Eikonal loss gradient: L = (||g||^2 - 1)^2
    const float g2 = gx*gx + gy*gy + gz*gz;
    const float dL_scale = 4.f * (g2 - 2.f) * weight ; // dL/dg = 4 (g^2-1) g
    const float dLdgx = dL_scale * gx;
    const float dLdgy = dL_scale * gy;
    const float dLdgz = dL_scale * gz;

    // 4) Chain rule to corner SDFs:
    // dL/dS[i] = dL/dg · ∂g/∂S[i] = (dLdgx*sx[i] + dLdgy*sy[i] + dLdgz*sz[i]) * (1/(4h))
    #pragma unroll
    for (int i=0;i<8;++i){
        const float dLdSi = (dLdgx * sx[i] + dLdgy * sy[i] + dLdgz * sz[i]) * 0.25f; //scaling originally inv4h
        atomicAdd(grid_pts_grad+ vox_key[v*8 + i] , dLdSi);
    }

}

// 2. C++ 
void voxel_gradient_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const torch::Tensor& vox_size_inv,
    const torch::Tensor& active_list,
    const float weight,
    const bool no_tv_s,
    const bool tv_sparse,
    const torch::Tensor& grid_pts_grad
) {
    // Launch CUDA kernel
    const int A = active_list.size(0);
    const int threads = 256;
    const int blocks = (A + threads - 1) / threads;
    voxel_gradient_kernel<<<blocks, threads>>>(
        grid_pts.contiguous().data_ptr<float>(),
        vox_key.contiguous().data_ptr<int64_t>(),
        vox_size_inv.contiguous().data_ptr<float>(),
        active_list.contiguous().data_ptr<int32_t>(),
        A,
        weight,
        grid_pts_grad.contiguous().data_ptr<float>()
    );
}

}
