
//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_ADD_ADD_H_
#define CUDATEST1_SRC_OPERATIONS_ADD_ADD_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include <iostream>

void add_host(
    const float* A,
    const float* B,
          float* C,
          unsigned int A_size,
          unsigned int B_size,
          unsigned int C_size,
          float alpha,
          float beta);

__global__ void add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
          unsigned int A_size,
          unsigned int B_size,
          unsigned int C_size,
          float alpha,
          float beta);

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
inline void add   ( const SArray<float> &A,
                    const SArray<float> &B,
                          SArray<float> &C,
                          float alpha,
                          float beta){


    if(mode == DEVICE){

        ASSERT(A.gpu_values);
        ASSERT(B.gpu_values);
        ASSERT(C.gpu_values);

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)C.size / block_size));
        add_kernel<<<grid, block>>>(
            A.gpu_values,
            B.gpu_values,
            C.gpu_values,
            A.size,
            B.size,
            C.size,
            alpha,
            beta);
//        cudaDeviceSynchronize();
    }else{
        add_host(
            A.cpu_values,
            B.cpu_values,
            C.cpu_values,
            A.size,
            B.size,
            C.size,
            alpha,
            beta);
    }
}

#endif //CUDATEST1_SRC_OPERATIONS_ADD_ADD_H_
