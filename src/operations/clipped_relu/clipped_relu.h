
//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_CLIPPED_RELU_H_
#define CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_CLIPPED_RELU_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include <iostream>
#include <cmath>

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

/**
 * performs C = A * alpha + B * beta
 * If A and B are not the same size as C, it will repeat the data contained in A and B
 * @tparam mode
 * @param A
 * @param B
 * @param C
 * @param alpha
 * @param beta
 */
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

#endif //CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_CLIPPED_RELU_H_
