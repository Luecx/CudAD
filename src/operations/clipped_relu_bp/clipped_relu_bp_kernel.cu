
//
// Created by Luecx on 13.01.2022.
//
#include "clipped_relu_bp.h"
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
__global__ void clipped_relu_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int size,
    float mmax){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size) return;

    if(A[idx] > 0 && A[idx] < mmax){
        A_grd[idx] = B_grd[idx];
    }else{
        A_grd[idx] = 0;
    }
}
