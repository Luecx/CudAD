
//
// Created by Luecx on 20.06.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MERGE_MERGE_H_
#define CUDAD_SRC_OPERATIONS_MERGE_MERGE_H_

#include "../../data/DenseMatrix.h"
#include "../../data/Mode.h"
#include "../../misc/config.h"

// clang-format off
void merge_host(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    unsigned int A_m,
    unsigned int B_m,
    unsigned int n);

__global__ void merge_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    unsigned int A_m,
    unsigned int B_m,
    unsigned int n);

template<Mode mode>
void merge(const DenseMatrix &A,
           const DenseMatrix &B,
                 DenseMatrix &C){

    ASSERT(A.m + B.m == C.m);
    ASSERT(A.n == B.n);
    ASSERT(A.n == C.n);

    ASSERT(A.address<mode>())
    ASSERT(B.address<mode>())
    ASSERT(C.address<mode>())

    if constexpr(mode == DEVICE){
        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;
        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)C.n / block_size_x),
                   std::ceil((float)C.m / block_size_y));
        merge_kernel<<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            C.gpuAddress(),
            A.m,
            B.m,
            C.n);
    }else{
        ERROR(false);
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_MERGE_MERGE_H_
