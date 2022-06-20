//
// Created by Luecx on 20.06.2022.
//
#include "merge.h"

__global__ void merge_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    unsigned int A_m,
    unsigned int B_m,
    unsigned int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int ldc = A_m + B_m;

    if (idx >= n || idy >= (ldc))
        return;

    if (idy < A_m){
        C[MATRIX_INDEX(ldc, idy, idx)] = A[MATRIX_INDEX(A_m, idy, idx)];
    }else{
        C[MATRIX_INDEX(ldc, idy, idx)] = B[MATRIX_INDEX(B_m, idy - A_m, idx)];
    }
}
