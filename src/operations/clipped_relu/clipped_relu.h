
//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_CLIPPED_RELU_H_
#define CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_CLIPPED_RELU_H_

#include "../../data/SArray.h"
#include "../../data/mode.h"

#include <cmath>
#include <iostream>

// clang-format off
void clipped_relu_host(
    const float* A,
          float* B,
    unsigned int size,
    float max);

__global__ void clipped_relu_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size,
    float max);

template<Mode mode>
inline void clipped_relu   (const SArray<float> &A,
                                  SArray<float> &B,
                                  float max){

    ASSERT(A.size == B.size)

    if(mode == DEVICE){

        ASSERT(A.gpu_values);
        ASSERT(B.gpu_values);

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)A.size / block_size));
        clipped_relu_kernel<<<grid, block>>>(
            A.gpu_values,
            B.gpu_values,
            A.size,
            max);
    }else{
        clipped_relu_host(
            A.cpu_values,
            B.cpu_values,
            A.size,
            max);
    }
}
// clang-format on

#endif    // CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_CLIPPED_RELU_H_
