
//
// Created by Luecx on 14.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_BP_H_
#define CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_BP_H_

#include "../../data/Matrix.h"
#include "../../data/DenseMatrix.h"
#include "../../data/SparseInput.h"
#include "../../data/mode.h"

__global__ void sparse_affine_bp_kernel(
          float*        __restrict__ mat_grd,
    const unsigned int* __restrict__ inp_col_indices,
    const unsigned int               inp_col_max_entries,
          float*        __restrict__ bia_grd,
    const float*        __restrict__ res_grd,
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
inline void sparse_affine_bp(
                   DenseMatrix& mat_grd,
                   SparseInput& inp,
                   DenseMatrix& bia_grd,
                   DenseMatrix& res_grd){

    auto M = mat_grd.m;
    auto N = mat_grd.n;
    auto B = inp.n;

    ASSERT(bia_grd.m == M)
    ASSERT(bia_grd.n == 1)
    ASSERT(inp.m     == N)
    ASSERT(res_grd.m == M)
    ASSERT(res_grd.n == B)

    if(mode == DEVICE){

        ASSERT(mat_grd.gpu_values)
        ASSERT(inp    .column_indices.gpu_values)
        ASSERT(bia_grd.gpu_values)
        ASSERT(res_grd.gpu_values)

//        mat_grd.clear<DEVICE>();
//        bia_grd.clear<DEVICE>();

        constexpr int block_size_x = 1;
        constexpr int block_size_y = 512;

        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)res_grd.n / block_size_x),
                   std::ceil((float)res_grd.m / block_size_y));

        sparse_affine_bp_kernel<<<grid, block>>>(
            mat_grd.gpu_values,
            inp.column_indices.gpu_values,
            inp.max_entries_per_column,
            bia_grd.gpu_values,
            res_grd.gpu_values,
            M,B,
            mat_grd.leading_dimension,
            res_grd.leading_dimension);
    }else{
        ASSERT(false)
    }
}

#endif //CUDAD_SRC_OPERATIONS_S_MM_SPARSE_AFFINE_BP_H_
