//
// Created by Luecx on 20.06.2022.
//
#include "merge_bp.h"

__global__ void merge_bp_kernel(
          float* __restrict__ A_grd,
          float* __restrict__ B_grd,
    const float* __restrict__ C_grd,
    unsigned int A_m,
    unsigned int B_m,
    unsigned int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int ldc = A_m + B_m;

    if (idx >= n || idy >= (ldc))
        return;

    if (idy < A_m){
        A_grd[MATRIX_INDEX(A_m, idy, idx)] = C_grd[MATRIX_INDEX(ldc, idy, idx)];
    }else{
        B_grd[MATRIX_INDEX(B_m, idy - A_m, idx)] = C_grd[MATRIX_INDEX(ldc, idy, idx)];
    }
}
