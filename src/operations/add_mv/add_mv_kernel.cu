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

#include "add_mv.h"

// clang-format off
__global__ void add_mv_kernel(
    const float* __restrict__ mat,
    const float* __restrict__ vec,
          float* __restrict__ res,
          int m,
          int n,
          int ld_mat,
          int ld_res){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    res[MATRIX_INDEX(ld_res, idy, idx)] = mat[MATRIX_INDEX(ld_mat, idy, idx)] + vec[idy];
}
