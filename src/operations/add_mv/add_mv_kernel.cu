
//
// Created by Luecx on 13.01.2022.
//
#include "add_mv.h"

// clang-format off
__global__ void add_mv_kernel(
    const float* __restrict__ mat,
    const float* __restrict__ vec,
          float* __restrict__ res,
          int m,
          int n,
          int ld_mat,
          int ld_res){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    res[MATRIX_INDEX(ld_res, idy, idx)] = mat[MATRIX_INDEX(ld_mat, idy, idx)] + vec[idy];
}
