
//
// Created by Luecx on 13.01.2022.
//
#include "sigmoid.h"
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
__global__ void sigmoid_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size,
    float scalar){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size) return;

    B[idx] = 1.0f / (1.0f + expf(- A[idx] * scalar));
}
