
//
// Created by Luecx on 20.06.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MERGE_BP_MERGE_BP_H_
#define CUDAD_SRC_OPERATIONS_MERGE_BP_MERGE_BP_H_

#include "../../data/DenseMatrix.h"
#include "../../data/Mode.h"
#include "../../misc/config.h"

// clang-format off
void merge_bp_host(
          float* __restrict__ A,
          float* __restrict__ B,
    const float* __restrict__ C,
    unsigned int A_m,
    unsigned int B_m,
    unsigned int n);

__global__ void merge_bp_kernel(
          float* __restrict__ A,
          float* __restrict__ B,
    const float* __restrict__ C,
    unsigned int A_m,
    unsigned int B_m,
    unsigned int n);

template<Mode mode>
void merge_bp(const DenseMatrix &A_grd,
              const DenseMatrix &B_grd,
                    DenseMatrix &C_grd){

    ASSERT(A_grd.m + B_grd.m == C_grd.m);
    ASSERT(A_grd.n == B_grd.n);
    ASSERT(A_grd.n == C_grd.n);

    ASSERT(A_grd.address<mode>())
    ASSERT(B_grd.address<mode>())
    ASSERT(C_grd.address<mode>())

    if constexpr(mode == DEVICE){
        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;
        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)C_grd.n / block_size_x),
                   std::ceil((float)C_grd.m / block_size_y));
        merge_bp_kernel<<<grid, block>>>(
            A_grd.gpuAddress(),
            B_grd.gpuAddress(),
            C_grd.gpuAddress(),
            A_grd.m,
            B_grd.m,
            C_grd.n);
    }else{
        ERROR(false);
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_MERGE_BP_MERGE_BP_H_
