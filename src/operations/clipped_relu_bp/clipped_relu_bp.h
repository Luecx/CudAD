
//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_BP_CLIPPED_RELU_BP_H_
#define CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_BP_CLIPPED_RELU_BP_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include <iostream>

void clipped_relu_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    unsigned int size,
    float max);

__global__ void clipped_relu_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
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
inline void clipped_relu_bp (const SArray<float> &A,
                                   SArray<float> &A_grd,
                             const SArray<float> &B,
                             const SArray<float> &B_grd,
                             float max){

    ASSERT(A.size == B.size)

    if(mode == DEVICE){

        ASSERT(A.gpu_values);
        ASSERT(B.gpu_values);

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)A.size / block_size));
        clipped_relu_bp_kernel<<<grid, block>>>(
            A    .gpu_values,
            A_grd.gpu_values,
            B    .gpu_values,
            B_grd.gpu_values,
            A.size,
            max);
    }else{
        clipped_relu_bp_host(
            A    .cpu_values,
            A_grd.cpu_values,
            B    .cpu_values,
            B_grd.cpu_values,
            A.size,
            max);
    }
}

#endif //CUDATEST1_SRC_OPERATIONS_CLIPPED_RELU_BP_CLIPPED_RELU_BP_H_
