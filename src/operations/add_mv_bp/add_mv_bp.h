
//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_ADD_MV_BP_ADD_MV_BP_H_
#define CUDATEST1_SRC_OPERATIONS_ADD_MV_BP_ADD_MV_BP_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include "../../data/DenseMatrix.h"
#include <iostream>

void add_mv_bp_host(
          float* mat_grd,
          float* vec_grd,
    const float* res_grd,
          int m,
          int n,
          int ld_mat,
          int ld_res);

__global__ void add_mv_bp_kernel(
          float* __restrict__ mat_grd,
          float* __restrict__ vec_grd,
    const float* __restrict__ res_grd,
          int m,
          int n,
          int ld_mat,
          int ld_res);

/**
 * @tparam mode
 * @param A
 * @param B
 * @param C
 * @param alpha
 * @param beta
 */
template<Mode mode>
inline void add_mv_bp( const DenseMatrix &mat_grd,
                       const DenseMatrix &vec_grd,
                             DenseMatrix &res_grd){

    ASSERT(mat_grd.m == res_grd.m);
    ASSERT(mat_grd.n == res_grd.n);
    ASSERT(mat_grd.m == vec_grd.m);
    ASSERT(vec_grd.n == 1);

    if(mode == DEVICE){

        ASSERT(mat_grd.gpu_values);
        ASSERT(vec_grd.gpu_values);
        ASSERT(res_grd.gpu_values);

        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;
        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)mat_grd.n / block_size_x),
                   std::ceil((float)mat_grd.m / block_size_y));
        add_mv_bp_kernel<<<grid, block>>>(
            mat_grd.gpu_values,
            vec_grd.gpu_values,
            res_grd.gpu_values,
            mat_grd.m,
            mat_grd.n,
            mat_grd.leading_dimension,
            res_grd.leading_dimension);
    }else{
        ASSERT(mat_grd.cpu_values);
        ASSERT(vec_grd.cpu_values);
        ASSERT(res_grd.cpu_values);

        add_mv_bp_host(
            mat_grd.cpu_values,
            vec_grd.cpu_values,
            res_grd.cpu_values,
            mat_grd.m,
            mat_grd.n,
            mat_grd.leading_dimension,
            res_grd.leading_dimension);
    }
}

#endif //CUDATEST1_SRC_OPERATIONS_ADD_MV_BP_ADD_MV_BP_H_
