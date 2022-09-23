//
// Created by Luecx on 02.07.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_SOFTMAX_SOFTMAX_H_
#define CUDAD_SRC_OPERATIONS_SOFTMAX_SOFTMAX_H_

#include "../../data/DenseMatrix.h"
#include "../../data/Mode.h"

// clang-format off
//void softmax_host(
//    const float* A,
//          float* B,
//    unsigned int m,
//    unsigned int n);

__global__ void softmax_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int m,
    unsigned int n);

template<Mode mode>
inline void softmax   (const DenseMatrix &A,
                             DenseMatrix &B){

    ASSERT(A.size() == B.size());
    ASSERT(A.n == B.n);

    if(mode == DEVICE){

        ASSERT(A.gpuAddress());
        ASSERT(B.gpuAddress());

        constexpr int block_size_m = 1;
        constexpr int block_size_n = 1024;
        dim3 block(block_size_n, block_size_m);
        dim3 grid (std::ceil((float)A.n / block_size_n),1);
        softmax_kernel<<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            A.m,
            A.n);
    }else{
//        softmax_host(
//            A.cpuAddress(),
//            B.cpuAddress(),
//            A.m,
//            A.n);
        ERROR(false);
    }
}
// clang-format on

#endif    // CUDAD_SRC_OPERATIONS_SOFTMAX_SOFTMAX_H_
