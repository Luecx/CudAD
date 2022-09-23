/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUDATEST1_SRC_OPERATIONS_ADD_MV_ADD_MV_H_
#define CUDATEST1_SRC_OPERATIONS_ADD_MV_ADD_MV_H_

#include "../../data/DenseMatrix.h"
#include "../../data/SArray.h"
#include "../../data/Mode.h"

#include <iostream>

// clang-format off
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

        ASSERT(mat.gpuAddress());
        ASSERT(vec.gpuAddress());
        ASSERT(res.gpuAddress());

        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;
        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)mat.n / block_size_x),
                   std::ceil((float)mat.m / block_size_y));
        add_mv_kernel<<<grid, block>>>(
            mat.gpuAddress(),
            vec.gpuAddress(),
            res.gpuAddress(),
            mat.m,
            mat.n,
            mat.leading_dimension,
            res.leading_dimension);
    }else{
        ASSERT(mat.cpuAddress());
        ASSERT(vec.cpuAddress());
        ASSERT(res.cpuAddress());

        add_mv_host(
            mat.cpuAddress(),
            vec.cpuAddress(),
            res.cpuAddress(),
            mat.m,
            mat.n,
            mat.leading_dimension,
            res.leading_dimension);
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_ADD_MV_ADD_MV_H_
