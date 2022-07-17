//
// Created by Luecx on 02.07.2022.
//
#include "softmax_bp.h"

__global__ void softmax_bp_kernel(
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int m,
    unsigned int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n){
        return;
    }
    if (idy >= m){
        return;
    }


    int offset = MATRIX_INDEX(m, 0, idx);
    int j = idy;

    float gradient = 0;
    for (int i = 0; i < m; i++){
        if(i == j){
            gradient += B_grd[offset + i] * B[offset + i] * (1 - B[offset + i]);
        }else{
            gradient += B_grd[offset + i] * B[offset + j] * (0 - B[offset + i]);
        }
    }
    A_grd[offset + j] = gradient;




}
