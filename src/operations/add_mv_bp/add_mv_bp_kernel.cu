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

#include "add_mv_bp.h"
// clang-format off
/**
 * performs C = alpha * A + beta * B
 * @param A
 * @param B
 * @param C
 * @param size
 * @param alpha
 * @param beta
 * @return
 */
__global__ void add_mv_bp_kernel(
          float* __restrict__ mat_grd,
          float* __restrict__ vec_grd,
    const float* __restrict__ res_grd,
          int m,
          int n,
          int ld_mat,
          int ld_res){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    float res_grd_v = res_grd[MATRIX_INDEX(ld_res, idy, idx)];

    if (res_grd_v == 0)
        return;

    mat_grd[MATRIX_INDEX(ld_mat, idy, idx)] = res_grd_v;

    atomicAdd(&vec_grd[idy], res_grd_v);
}
