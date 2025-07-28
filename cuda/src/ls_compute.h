#ifndef LS_COMPUTE_H_INCLUDED
#define LS_COMPUTE_H_INCLUDED

#include <torch/extension.h>

namespace LS_COMPUTE {

// Python interface to directly write the gradient of tv loss.
void laplacian_smoothness_bw(
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
    const torch::Tensor& grid_pts_grad);

}

#endif
