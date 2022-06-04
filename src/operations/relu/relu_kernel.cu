
//
// Created by Luecx on 13.01.2022.
//
#include "relu.h"

// clang-format off
__global__ void relu_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    B[idx] = max(0.0f, A[idx]);
}
