/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef VG_COMPUTE_H_INCLUDED
#define VG_COMPUTE_H_INCLUDED

#include <torch/extension.h>

namespace VG_COMPUTE {

// Python interface to directly write the gradient of tv loss.
void voxel_gradient_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const torch::Tensor& vox_size_inv,
    const torch::Tensor& active_list,
    const float weight,
    const bool no_tv_s,
    const bool tv_sparse,
    const torch::Tensor& grid_pts_grad
) ;

} 

#endif
