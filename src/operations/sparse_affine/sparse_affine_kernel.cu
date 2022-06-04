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

#include "sparse_affine.h"
// clang-format off
__global__ void sparse_affine_kernel(
    const float*        __restrict__ mat,
    const unsigned int* __restrict__ inp_col_indices,
    const unsigned int               inp_col_max_entries,
    const float*        __restrict__ bia,
          float*        __restrict__ res,
    const unsigned int               m,
    const unsigned int               n,
    const unsigned int               lda,
    const unsigned int               ldc){

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
    float sum = bia[row];

    // start at offset + 1 (offset contains the amount of values to read)
    for (int i = offset + 1; i < offset + 1 + count; i++) {

        // get the sparse index (set row of the input)
        auto b_row = inp_col_indices[i];
        // get the corresponding weight
        auto wgt = mat[MATRIX_INDEX(lda, row, b_row)];
        sum += wgt;
    }
    res[MATRIX_INDEX(ldc, row, col)] = sum;
};