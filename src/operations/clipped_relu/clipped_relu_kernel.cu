
//
// Created by Luecx on 13.01.2022.
//
#include "clipped_relu.h"
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
__global__ void clipped_relu_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size,
    float mmax){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size) return;

    B[idx] = min(mmax, max(0, A[idx]));
}
