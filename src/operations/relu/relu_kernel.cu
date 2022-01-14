
//
// Created by Luecx on 13.01.2022.
//
#include "relu.h"
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
__global__ void relu_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size) return;

    B[idx] = max(0.0f, A[idx]);
}
