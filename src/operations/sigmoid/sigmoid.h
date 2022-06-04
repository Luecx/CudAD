
//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_SIGMOID_SIGMOID_H_
#define CUDATEST1_SRC_OPERATIONS_SIGMOID_SIGMOID_H_

#include "../../data/SArray.h"
#include "../../data/mode.h"

#include <iostream>

// clang-format off
void sigmoid_host(
    const float* A,
          float* B,
    unsigned int size,
    float scalar);

__global__ void sigmoid_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size,
    float scalar);

template<Mode mode>
inline void sigmoid   (const SArray<float> &A,
                             SArray<float> &B,
                             float scalar){

    ASSERT(A.size == B.size)

    if(mode == DEVICE){

        ASSERT(A.gpu_values);
        ASSERT(B.gpu_values);

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)A.size / block_size));
        sigmoid_kernel<<<grid, block>>>(
            A.gpu_values,
            B.gpu_values,
            A.size,
            scalar);
    }else{
        sigmoid_host(
            A.cpu_values,
            B.cpu_values,
            A.size,
            scalar);
    }
}
// clang-format on

#endif    // CUDATEST1_SRC_OPERATIONS_SIGMOID_SIGMOID_H_
