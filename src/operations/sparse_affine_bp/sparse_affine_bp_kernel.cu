
//
// Created by Luecx on 14.01.2022.
//

#include "sparse_affine_bp.h"
__global__ void sparse_affine_bp_kernel(
          float*        __restrict__ mat_grd,
    const unsigned int* __restrict__ inp_col_indices,
    const unsigned int               inp_col_max_entries,
          float*        __restrict__ bia_grd,
    const float*        __restrict__ res_grd,
    const unsigned int               m,
    const unsigned int               n,
    const unsigned int               lda,
    const unsigned int               ldc){

    // compute which output value we are looking at
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // skip out of bounds
    if (col >= n || row >= m) return;

    // get the offset at which we look into our sparse input
    int offset = col * (inp_col_max_entries + 1);
    // check how many values we are going to read
    int count = inp_col_indices[offset];

    // track the sum
    float res_grd_v = res_grd[MATRIX_INDEX(ldc, row, col)];

    atomicAdd(&bia_grd[row], res_grd_v);

    // start at offset + 1 (offset contains the amount of values to read)
    for (int i = offset + 1; i < offset + 1 + count; i++) {

        // get the sparse index (set row of the input)
        auto b_row = inp_col_indices[i];
        // get the corresponding weight
        atomicAdd(&mat_grd[MATRIX_INDEX(lda, row, b_row)], res_grd_v);
    }
};