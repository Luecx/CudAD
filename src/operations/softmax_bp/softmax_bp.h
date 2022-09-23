//
// Created by Luecx on 02.07.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_SOFTMAX_BP_SOFTMAX_BP_H_
#define CUDAD_SRC_OPERATIONS_SOFTMAX_BP_SOFTMAX_BP_H_

#include "../../data/DenseMatrix.h"
#include "../../data/Mode.h"

// clang-format off
//void softmax_host(
//    const float* A,
//          float* B,
//    unsigned int m,
//    unsigned int n);

__global__ void softmax_bp_kernel(
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int m,
    unsigned int n);

template<Mode mode>
inline void softmax_bp(      DenseMatrix &A_grd,
                       const DenseMatrix &B,
                       const DenseMatrix &B_grd){

    ASSERT(A_grd.size() == B.size());
    ASSERT(A_grd.n == B.n);

    if(mode == DEVICE){

        ASSERT(A_grd.gpuAddress());
        ASSERT(B.gpuAddress());
        ASSERT(B_grd.gpuAddress());

        constexpr int block_size_m = 16;
        constexpr int block_size_n = 16;
        dim3 block(block_size_n, block_size_m);
        dim3 grid (std::ceil((float)A_grd.n / block_size_n),std::ceil((float)A_grd.m / block_size_m));
        softmax_bp_kernel<<<grid, block>>>(
            A_grd.gpuAddress(),
            B.gpuAddress(),
            B_grd.gpuAddress(),
            A_grd.m,
            A_grd.n);
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

#endif    // CUDAD_SRC_OPERATIONS_SOFTMAX_BP_SOFTMAX_BP_H_
