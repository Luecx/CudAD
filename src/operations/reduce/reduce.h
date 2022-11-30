
//
// Created by Luecx on 20.06.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_reduce_reduce_H_
#define CUDAD_SRC_OPERATIONS_reduce_reduce_H_

#include "../../data/DenseMatrix.h"
#include "../../data/Mode.h"
#include "../../misc/config.h"

// clang-format off
//void reduce_host(
//    const float* __restrict__ A,
//    const float* __restrict__ B,
//          float* __restrict__ C,
//    unsigned int A_m,
//    unsigned int B_m,
//    unsigned int n);

__global__ void reduce_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ wgt,
          float* __restrict__ out,
    unsigned int inp_size,
    unsigned int out_size,
    unsigned int bat_size,
    unsigned int reduction);

template<Mode mode>
void reduce(const DenseMatrix &inp,
            const DenseMatrix &wgt,
                  DenseMatrix &out){

    ASSERT(inp.address<mode>())
    ASSERT(wgt.address<mode>())
    ASSERT(out.address<mode>())

    ASSERT(inp.n == out.n);
    ASSERT(inp.m == wgt.m);

    if constexpr(mode == DEVICE){
        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;

        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)inp.n / block_size_x),
                   std::ceil((float)out.m / block_size_y));
        reduce_kernel<<<grid, block>>>(
            inp.gpuAddress(),
            wgt.gpuAddress(),
            out.gpuAddress(),
            inp.m,
            out.m,
            inp.n,
            inp.m / out.m);
    }else{
        ERROR(false);
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_reduce_reduce_H_
