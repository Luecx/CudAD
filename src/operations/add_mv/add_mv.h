
//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_ADD_MV_ADD_MV_H_
#define CUDATEST1_SRC_OPERATIONS_ADD_MV_ADD_MV_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include "../../data/DenseMatrix.h"
#include <iostream>

void add_mv_host(
    const float* mat,
    const float* vec,
          float* res,
          int m,
          int n,
          int ld_mat,
          int ld_res);

__global__ void add_mv_kernel(
    const float* __restrict__ mat,
    const float* __restrict__ vec,
          float* __restrict__ res,
          int m,
          int n,
          int ld_mat,
          int ld_res);

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
inline void add_mv( const DenseMatrix &mat,
                    const DenseMatrix &vec,
                          DenseMatrix &res){

    ASSERT(mat.m == res.m);
    ASSERT(mat.n == res.n);
    ASSERT(mat.m == vec.m);
    ASSERT(vec.n == 1);

    if(mode == DEVICE){

        ASSERT(mat.gpu_values);
        ASSERT(vec.gpu_values);
        ASSERT(res.gpu_values);

        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;
        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)mat.n / block_size_x),
                   std::ceil((float)mat.m / block_size_y));
        add_mv_kernel<<<grid, block>>>(
            mat.gpu_values,
            vec.gpu_values,
            res.gpu_values,
            mat.m,
            mat.n,
            mat.leading_dimension,
            res.leading_dimension);
    }else{
        ASSERT(mat.cpu_values);
        ASSERT(vec.cpu_values);
        ASSERT(res.cpu_values);

        add_mv_host(
            mat.cpu_values,
            vec.cpu_values,
            res.cpu_values,
            mat.m,
            mat.n,
            mat.leading_dimension,
            res.leading_dimension);
    }
}

#endif //CUDATEST1_SRC_OPERATIONS_ADD_MV_ADD_MV_H_
