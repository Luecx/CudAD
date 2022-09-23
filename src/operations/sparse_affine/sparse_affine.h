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

#ifndef CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_H_
#define CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_H_

#include "../../data/DenseMatrix.h"
#include "../../data/Matrix.h"
#include "../../data/SparseInput.h"
#include "../../data/Mode.h"

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
    const unsigned int               ldc);

/**
 * Performs
 * networks = matrix * inp + bia
 *
 * Dimensions:
 * matrix = [M,N]
 * bias   = [M,1]
 * inp    = [N,B]
 * networks    = [M,B]
 *
 * where B = BATCH_SIZE
 *       M = OUTPUT SIZE
 *       N = INPUT SIZE
 *
 * @param matrix
 * @param inp
 * @param bia
 * @param res
 */
template<Mode mode>
inline void sparse_affine(
                   DenseMatrix& mat,
                   SparseInput& inp,
                   DenseMatrix& bia,
                   DenseMatrix& res){

    auto M = mat.m;
    [[maybe_unused]]
    auto N = mat.n;
    auto B = inp.n;

    ASSERT(bia.m == M)
    ASSERT(bia.n == 1)
    ASSERT(inp.m == N)
    ASSERT(res.m == M)
    ASSERT(res.n == B)

    if(mode == DEVICE){

        ASSERT(mat.gpuAddress())
        ASSERT(inp.column_indices.gpuAddress())
        ASSERT(bia.gpuAddress())
        ASSERT(res.gpuAddress())

        constexpr int block_size_x = 1;
        constexpr int block_size_y = 256;

        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)res.n / block_size_x),
                   std::ceil((float)res.m / block_size_y));

        sparse_affine_kernel<<<grid, block>>>(
            mat.gpuAddress(),
            inp.column_indices.gpuAddress(),
            inp.max_entries_per_column,
            bia.gpuAddress(),
            res.gpuAddress(),
            M,B,
            mat.leading_dimension,
            res.leading_dimension);
    }else{
        ASSERT(false)
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_H_
