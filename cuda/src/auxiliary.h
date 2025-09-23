/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef RASTERIZER_AUXILIARY_H_INCLUDED
#define RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"

// Octant ordering tables
template<uint64_t id, int n>
struct repeat_3bits
{
    static constexpr uint64_t value = (repeat_3bits<id, n-1>::value << 3) | id;
};

template<uint64_t id>
struct repeat_3bits<id, 1>
{
    static constexpr uint64_t value = id;
};

__constant__ uint64_t order_tables[8] = {
    repeat_3bits<0ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<1ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<2ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<3ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<4ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<5ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<6ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<7ULL, MAX_NUM_LEVELS>::value
};

__forceinline__ __device__ uint64_t compute_order_rank(uint64_t octree_path, int quadrant_id)
{
    return octree_path ^ order_tables[quadrant_id];
}

__forceinline__ __device__ uint64_t encode_order_key(uint64_t tile_id, uint64_t order_rank)
{
    return (tile_id << NUM_BIT_ORDER_RANK) | order_rank;
}

__forceinline__ __device__ uint32_t encode_order_val(uint32_t vox_id, uint32_t quadrant_id)
{
    return (((uint32_t)quadrant_id) << 29) | vox_id;
}

__forceinline__ __device__ uint32_t decode_order_val_4_vox_id(uint32_t val)
{
    return (val << 3) >> 3;
}

__forceinline__ __device__ uint32_t decode_order_val_4_quadrant_id(uint32_t val)
{
    return val >> 29;
}

__forceinline__ __device__ uint32_t compute_ray_quadrant_id(float3 rd)
{
    return ((rd.x < 0) << 2) | ((rd.y < 0) << 1) | (rd.z < 0);
}

__forceinline__ __device__ uint32_t compute_corner_quadrant_id(float3 corner, float3 ro)
{
    return ((corner.x < ro.x) << 2) | ((corner.y < ro.y) << 1) | (corner.z < ro.z);
}

// Utility functions.
__forceinline__ __device__ float3 compute_ray_d(float2 pixf, int W, int H, float tan_fovx, float tan_fovy, float cx, float cy)
{
    const float3 rd = {
        (pixf.x + 0.5f - cx) * 2.f * tan_fovx / (float)W,
        (pixf.y + 0.5f - cy) * 2.f * tan_fovy / (float)H,
        1.f
    };
    return rd;
}

__forceinline__ __device__ void getBboxTileRect(const uint2& bbox, uint2& tile_min, uint2& tile_max, dim3 grid)
{
    uint32_t xmin = (bbox.x >> 16);
    uint32_t ymin = (bbox.x << 16 >> 16);
    uint32_t xmax = (bbox.y >> 16);
    uint32_t ymax = (bbox.y << 16 >> 16);
    tile_min = {
        (uint32_t)max(0, min(((int)grid.x)-1, (int)(xmin / BLOCK_X))),
        (uint32_t)max(0, min(((int)grid.y)-1, (int)(ymin / BLOCK_Y)))
    };
    tile_max = {
        (uint32_t)max(0, min(((int)grid.x)-1, (int)(xmax / BLOCK_X))),
        (uint32_t)max(0, min(((int)grid.y)-1, (int)(ymax / BLOCK_Y)))
    };
}

__forceinline__ __device__ bool pix_in_bbox(const uint2& pix, const uint2& bbox)
{
    bool valid_xmin = pix.x >= (bbox.x >> 16);
    bool valid_ymin = pix.y >= (bbox.x << 16 >> 16);
    bool valid_xmax = pix.x <= (bbox.y >> 16);
    bool valid_ymax = pix.y <= (bbox.y << 16 >> 16);
    return valid_xmin && valid_ymin && valid_xmax && valid_ymax;
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(a.x * b, a.y * b);
}

__forceinline__ __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

__forceinline__ __device__ float2 min(const float2& a, const float2& b)
{
    return make_float2(min(a.x, b.x), min(a.y, b.y));
}

__forceinline__ __device__ float2 max(const float2& a, const float2& b)
{
    return make_float2(max(a.x, b.x), max(a.y, b.y));
}

__forceinline__ __device__ float3 clamp0(const float3& a)
{
    return make_float3(max(a.x, 0.f), max(a.y, 0.f), max(a.z, 0.f));
}

__forceinline__ __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 last_col_3x4(const float* matrix)
{
    float3 last_col = {matrix[3], matrix[7], matrix[11]};
    return last_col;
}

__forceinline__ __device__ float3 third_col_3x4(const float* matrix)
{
    float3 third_col = {matrix[2], matrix[6], matrix[10]};
    return third_col;
}


__forceinline__ __device__ float3 transform_3x4(const float* matrix, const float3& p)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11]
    };
    return transformed;
}

__forceinline__ __device__ float3 rotate_3x4(const float* matrix, const float3& p)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z
    };
    return transformed;
}

// Depth convertion
__forceinline__ __device__ float depth_contracted(float x)
{
    return (x < 1.f) ? x : 2.f - 1.f / x;
}

__forceinline__ __device__ float2 ray_aabb(float3 vox_c, float vox_l, float3 ro, float3 rd_inv)
{
    float vox_r = 0.5f * vox_l;
    float3 dir = vox_c - ro;
    float3 c0_ = (dir - vox_r) * rd_inv;
    float3 c1_ = (dir + vox_r) * rd_inv;
    float3 c0 = make_float3(min(c0_.x, c1_.x), min(c0_.y, c1_.y), min(c0_.z, c1_.z));
    float3 c1 = make_float3(max(c0_.x, c1_.x), max(c0_.y, c1_.y), max(c0_.z, c1_.z));
    float2 ab = make_float2(
        max(max(c0.x, c0.y), c0.z),
        min(min(c1.x, c1.y), c1.z)
    );
    return ab;
}

__forceinline__ __device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float softplus(float x)
{
    return (x > SOFTPLUS_THRES) ? x : log1pf(expf(x));
}

__forceinline__ __device__ float2 softplus2(float base, float x)
{
    const float total = base + x;
    return (total > SOFTPLUS_THRES)
                ? make_float2(base, x)
                : make_float2(log1pf(expf(total)), 0.0f);
}

__forceinline__ __device__ float softplus_bw(float x)
{
    return (x > SOFTPLUS_THRES) ? 1.f : sigmoid(x);
}

__forceinline__ __device__ float exp_linear_11(float x)
{
    return (x > 1.1f) ? x : expf(0.909090909091f * x - 0.904689820196f);
}

__forceinline__ __device__ float exp_linear_11_bw(float x)
{
    return (x > 1.1f) ? 1.0f : 0.909090909091f * expf(0.909090909091f * x - 0.904689820196f);
}

__forceinline__ __device__ float relu(float x)
{
    return max(x, 0.f);
}

__forceinline__ __device__ float relu_bw(float x)
{
    return float(x > 0.f);
}

__forceinline__ __device__ float safe_rnorm(const float3& v)
{
    return rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z + 1e-15f);
}

__forceinline__ __device__ float safe_rnorm(const float x, const float y, const float z)
{
    return rsqrtf(x*x + y*y + z*z + 1e-15f);
}

__forceinline__ __device__ float tri_interp_weight(const float3 qt, float interp_w[8])
{
    float wx[2] = {1.f - qt.x, qt.x};
    float wy[2] = {1.f - qt.y, qt.y};
    float wz[2] = {1.f - qt.z, qt.z};
    interp_w[0] = wx[0] * wy[0] * wz[0];
    interp_w[1] = wx[0] * wy[0] * wz[1];
    interp_w[2] = wx[0] * wy[1] * wz[0];
    interp_w[3] = wx[0] * wy[1] * wz[1];
    interp_w[4] = wx[1] * wy[0] * wz[0];
    interp_w[5] = wx[1] * wy[0] * wz[1];
    interp_w[6] = wx[1] * wy[1] * wz[0];
    interp_w[7] = wx[1] * wy[1] * wz[1];
    return interp_w[0] + interp_w[1] + interp_w[2] + interp_w[3] +
           interp_w[4] + interp_w[5] + interp_w[6] + interp_w[7];
}
__forceinline__ __device__ void tri_interp_grad_weight(
    const float3 qt,
    float grad_interp_w[8][3]  // [8 vertices][dx, dy, dz]
) {
    float x = qt.x, y = qt.y, z = qt.z;
    float dx[2] = {-1.f, 1.f};
    float dy[2] = {-1.f, 1.f};
    float dz[2] = {-1.f, 1.f};

    // Precompute weight components
    float wx[2] = {1.f - x, x};
    float wy[2] = {1.f - y, y};
    float wz[2] = {1.f - z, z};

    for (int ix = 0; ix <= 1; ++ix) {
        for (int iy = 0; iy <= 1; ++iy) {
            for (int iz = 0; iz <= 1; ++iz) {
                int idx = (ix << 2) | (iy << 1) | iz;
                grad_interp_w[idx][0] = dx[ix] * wy[iy] * wz[iz]; // ?ˆ‚w/?ˆ‚x
                grad_interp_w[idx][1] = wx[ix] * dy[iy] * wz[iz]; // ?ˆ‚w/?ˆ‚y
                grad_interp_w[idx][2] = wx[ix] * wy[iy] * dz[iz]; // ?ˆ‚w/?ˆ‚z
            }
        }
    }
}
__forceinline__ __device__ int flatten_grid_key(int x, int y, int z, int res) {
    return x + y * res + z * res * res;
}
__forceinline__ __device__ bool check_valid_neighbors(
    int x, int y, int z, int grid_res, const bool* __restrict__ grid_mask
) {
    if (x <= 0 || x >= grid_res - 1 ||
        y <= 0 || y >= grid_res - 1 ||
        z <= 0 || z >= grid_res - 1)
        return false;
    return grid_mask[flatten_grid_key(x+1, y, z, grid_res)] &&
           grid_mask[flatten_grid_key(x-1, y, z, grid_res)] &&
           grid_mask[flatten_grid_key(x, y+1, z, grid_res)] &&
           grid_mask[flatten_grid_key(x, y-1, z, grid_res)] &&
           grid_mask[flatten_grid_key(x, y, z+1, grid_res)] &&
           grid_mask[flatten_grid_key(x, y, z-1, grid_res)];
}
__forceinline__ __device__ int binary_search(const int32_t* keys, int size, int key) {
    int l = 0, r = size - 1;
    while (l <= r) {
        int m = (l + r) / 2;
        if (keys[m] == key) return m;
        if (keys[m] < key) l = m + 1;
        else r = m - 1;
    }
    return -1;
}


__forceinline__ __device__ float3 operator/(const float3& a, const float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__forceinline__ __device__ float interpolate_sdf_and_grad(
    int M, int gx, int gy, int gz,
    const int32_t* grid_keys, const int32_t* grid2voxel,
    const float* voxel_coords, const float* voxel_sizes,
    const int64_t* vox_key, const float* grid_pts, int grid_res, int num_voxels, int grid_pts_size,
    float* weights_out, int& voxel_id_out, float3 randv
) {
    int gk = flatten_grid_key(gx, gy, gz, grid_res);
    int idx = binary_search(grid_keys,M , gk);
    if (idx == -1) {
        printf("Error: grid key %d not found in grid_keys.\n", gk);
        return 0.0f;
    }
    voxel_id_out = grid2voxel[idx];
    float3 pt = make_float3(gx+randv.x, gy+randv.y, gz+randv.z);

    float3 base = make_float3(
        voxel_coords[voxel_id_out * 3 + 0],
        voxel_coords[voxel_id_out * 3 + 1],
        voxel_coords[voxel_id_out * 3 + 2]
    );
    //if(grid_res==128) printf("Voxel ID: %d, Base: (%f, %f, %f)\nVoxel ID: %d, Point: (%f, %f, %f)\n", voxel_id_out, base.x, base.y, base.z, voxel_id_out, pt.x, pt.y, pt.z);
    
    if (voxel_id_out < 0 || voxel_id_out >= num_voxels) {
        printf("Error: voxel_id_out %d is out of bounds.\n", voxel_id_out);
        return 0.0f;
    }
    float size = voxel_sizes[voxel_id_out];
    float3 local_q = (pt - base) / size;
    tri_interp_weight(local_q, weights_out);

    float sdf = 0.f;
    for (int i = 0; i < 8; ++i) {
        if(weights_out[i]<0) {
            weights_out[i] = 0;
            //printf("Error: weight[%d] = %f is out of bounds [0, 1].\n", i, weights_out[i]);
        }
        if(weights_out[i]>1 ){
            weights_out[i] = 1;
            //printf("Error: weight[%d] = %f is out of bounds [0, 1].\n", i, weights_out[i]);
        }
        int pt_id = vox_key[voxel_id_out * 8 + i];
        if (pt_id < 0 || pt_id >= grid_pts_size) {
            printf("Error: pt_id %d is out of bounds.\n", pt_id);
            return 0.0f;
        }
        sdf += grid_pts[pt_id] * weights_out[i];
    }
    return sdf;
}

__forceinline__ __device__ void accumulate_grad(
    int voxel_id, int num_voxels, const int64_t* vox_key,
    const float* w, float dL_dsdf, float* grid_pts_grad
) {
    if (voxel_id < 0 || voxel_id >= num_voxels) {
        printf("Error: voxel_id %d is out of bounds.\n", voxel_id);
        return;
    }
    for (int i = 0; i < 8; ++i) {
        int pt_id = vox_key[voxel_id * 8 + i];
        atomicAdd(grid_pts_grad + pt_id, dL_dsdf * w[i]);
    }
}
__forceinline__ __device__ float hash_to_float(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return (x & 0xFFFFFF) / float(0x1000000);  // ?ˆˆ [0,1)
}
__forceinline__ __device__ float3 get_rand_vec(int tid, int seed) {
    return make_float3(
        hash_to_float(seed * 73856093 ^ tid * 3 + 0),
        hash_to_float(seed * 19349663 ^ tid * 3 + 1),
        hash_to_float(seed * 83492791 ^ tid * 3 + 2)
    );
}
// Debugging helper.
#define CHECK_CUDA(debug) \
if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
