
//
// Created by Luecx on 13.01.2022.
//
#include "sigmoid.h"

// clang-format off
__global__ void sigmoid_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size,
    float scalar){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    B[idx] = 1.0f / (1.0f + expf(-A[idx] * scalar));
}
