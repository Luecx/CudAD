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

#ifndef CUDATEST1_SRC_OPERATIONS_ADD_MV_BP_ADD_MV_BP_H_
#define CUDATEST1_SRC_OPERATIONS_ADD_MV_BP_ADD_MV_BP_H_

#include "../../data/DenseMatrix.h"
#include "../../data/SArray.h"
#include "../../data/Mode.h"

#include <iostream>

// clang-format off
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

        ASSERT(mat_grd.gpuAddress());
        ASSERT(vec_grd.gpuAddress());
        ASSERT(res_grd.gpuAddress());

        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;
        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)mat_grd.n / block_size_x),
                   std::ceil((float)mat_grd.m / block_size_y));
        add_mv_bp_kernel<<<grid, block>>>(
            mat_grd.gpuAddress(),
            vec_grd.gpuAddress(),
            res_grd.gpuAddress(),
            mat_grd.m,
            mat_grd.n,
            mat_grd.leading_dimension,
            res_grd.leading_dimension);
    }else{
        ASSERT(mat_grd.cpuAddress());
        ASSERT(vec_grd.cpuAddress());
        ASSERT(res_grd.cpuAddress());

        add_mv_bp_host(
            mat_grd.cpuAddress(),
            vec_grd.cpuAddress(),
            res_grd.cpuAddress(),
            mat_grd.m,
            mat_grd.n,
            mat_grd.leading_dimension,
            res_grd.leading_dimension);
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_ADD_MV_BP_ADD_MV_BP_H_
