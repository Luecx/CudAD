
//
// Created by Luecx on 28.05.2022.
//
#include "pairwise_multiply_bp.h"

// clang-format off
__global__ void pairwise_multiply_bp_kernel(
    const float* __restrict__ input,
          float* __restrict__ input_grd,
    const float* __restrict__ output_grd,
    unsigned int outsize){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= outsize)
        return;

    int idx1        = idx * 2;
    int idx2        = idx1 + 1;

    input_grd[idx1] = output_grd[idx] * input[idx2];
    input_grd[idx2] = output_grd[idx] * input[idx1];
}