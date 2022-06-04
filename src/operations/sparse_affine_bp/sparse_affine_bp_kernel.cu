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

#include "sparse_affine_bp.h"
// clang-format off
__global__ void sparse_affine_bp_kernel(
          float*        __restrict__ mat_grd,
    const unsigned int* __restrict__ inp_col_indices,
    const unsigned int               inp_col_max_entries,
          float*        __restrict__ bia_grd,
    const float*        __restrict__ res,
    const float*        __restrict__ res_grd,
    const unsigned int               m,
    const unsigned int               n,
    const unsigned int               lda,
    const unsigned int               ldc,
          float                      lasso_regularization){
    // clang-format on

    // compute which output value we are looking at
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // skip out of bounds
    if (col >= n || row >= m)
        return;

    // get the offset at which we look into our sparse input
    int offset = col * (inp_col_max_entries + 1);
    // check how many values we are going to read
    int count = inp_col_indices[offset];

    // track the sum
    float res_grd_v = res_grd[MATRIX_INDEX(ldc, row, col)];

    // lasso
    //    res_grd_v += (1.0 / 8388608.0) * (res[MATRIX_INDEX(ldc, row, col)] > 0);
    res_grd_v += lasso_regularization * (res[MATRIX_INDEX(ldc, row, col)] > 0);

    // dont do anything if the gradient is 0. Theoretical impact on memory
    if (res_grd_v == 0)
        return;

    atomicAdd(&bia_grd[row], res_grd_v);

    // start at offset + 1 (offset contains the amount of values to read)
    for (int i = offset + 1; i < offset + 1 + count; i++) {

        // get the sparse index (set row of the input)
        auto b_row = inp_col_indices[i];
        // get the corresponding weight
        atomicAdd(&mat_grd[MATRIX_INDEX(lda, row, b_row)], res_grd_v);
    }
};