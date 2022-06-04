
//
// Created by Luecx on 13.01.2022.
//
#include "sigmoid_bp.h"

// clang-format off
__global__ void sigmoid_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int size,
    float scalar){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    A_grd[idx] = B_grd[idx] * B[idx] * (1 - B[idx]) * scalar;
}
