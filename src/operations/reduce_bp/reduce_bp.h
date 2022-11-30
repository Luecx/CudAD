
//
// Created by Luecx on 20.06.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_reduce_bp_reduce_bp_H_
#define CUDAD_SRC_OPERATIONS_reduce_bp_reduce_bp_H_

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

__global__ void reduce_bp_kernel(
    const float* __restrict__ inp,
          float* __restrict__ inp_grd,
    const float* __restrict__ wgt,
          float* __restrict__ wgt_grd,
    const float* __restrict__ out_grd,
    unsigned int inp_size,
    unsigned int out_size,
    unsigned int bat_size,
    unsigned int reduction);


template<Mode mode>
void reduce_bp(const DenseMatrix &inp,
                     DenseMatrix &inp_grd,
               const DenseMatrix &wgt,
                     DenseMatrix &wgt_grd,
               const DenseMatrix &out_grd){

//    ASSERT(inp.address<mode>())
//    ASSERT(inp_grd.address<mode>())
//    ASSERT(wgt.address<mode>())
//    ASSERT(wgt_wgt.address<mode>())
//    ASSERT(out_grd.address<mode>())
//
//    ASSERT(inp.m == inp_grd.m);
//    ASSERT(inp.n == inp_grd.n);
//
//    ASSERT(wgt.m == wgt_wgt.m);
//    ASSERT(wgt.n == wgt_wgt.n);
//
//    ASSERT(inp.n == inp_grd.n);
//    ASSERT(inp.n == wgt.n);
//    ASSERT(inp.n == wgt_wgt.n);
//    ASSERT(inp.n == out_grd.n);

    if constexpr(mode == DEVICE){
        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;

        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)inp_grd.n / block_size_x),
                   std::ceil((float)out_grd.m / block_size_y));
        reduce_bp_kernel<<<grid, block>>>(
            inp.gpu_values,
            inp_grd.gpu_values,
            wgt.gpu_values,
            wgt_grd.gpu_values,
            out_grd.gpu_values,
            inp.m,
            out_grd.m,
            inp.n,
            inp.m / out_grd.m);
    }else{
//        ERROR(false);
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_reduce_reduce_H_
