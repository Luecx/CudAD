
//
// Created by Luecx on 13.01.2022.
//
#include "add_mv_bp.h"
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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx >= n || idy >= m) return;


    float res_grd_v = res_grd[MATRIX_INDEX(ld_res, idy, idx)];

    if(res_grd_v == 0) return;

    mat_grd[MATRIX_INDEX(ld_mat, idy, idx)] = res_grd_v;

    atomicAdd(&vec_grd[idy], res_grd_v);
}
