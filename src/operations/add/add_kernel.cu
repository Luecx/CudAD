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

#include "add.h"
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
// clang-format off
__global__ void add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    const unsigned int A_size,
    const unsigned int B_size,
    const unsigned int C_size,
    const unsigned int size,
    const float alpha,
    const float beta){

    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    C[idx] = (A[idx % A_size] * alpha + B[idx % B_size] * beta);
}
