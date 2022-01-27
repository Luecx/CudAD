
//
// Created by Luecx on 14.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_H_
#define CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_H_

#include "../../data/Matrix.h"
#include "../../data/DenseMatrix.h"
#include "../../data/SparseInput.h"
#include "../../data/mode.h"

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
    auto N = mat.n;
    auto B = inp.n;

    ASSERT(bia.m == M)
    ASSERT(bia.n == 1)
    ASSERT(inp.m == N)
    ASSERT(res.m == M)
    ASSERT(res.n == B)

    if(mode == DEVICE){

        ASSERT(mat.gpu_values)
        ASSERT(inp.column_indices.gpu_values)
        ASSERT(bia.gpu_values)
        ASSERT(res.gpu_values)

        constexpr int block_size_x = 1;
        constexpr int block_size_y = 256;

        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)res.n / block_size_x),
                   std::ceil((float)res.m / block_size_y));

        sparse_affine_kernel<<<grid, block>>>(
            mat.gpu_values,
            inp.column_indices.gpu_values,
            inp.max_entries_per_column,
            bia.gpu_values,
            res.gpu_values,
            M,B,
            mat.leading_dimension,
            res.leading_dimension);
    }else{
        ASSERT(false)
    }
}

#endif //CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_H_
