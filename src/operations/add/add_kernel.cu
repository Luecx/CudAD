
//
// Created by Luecx on 13.01.2022.
//
#include "add.h"
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
__global__ void add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    const unsigned int A_size,
    const unsigned int B_size,
    const unsigned int C_size,
    const float alpha,
    const float beta){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= C_size) return;

    C[idx] = A[idx % A_size] * alpha + B[idx % B_size] * beta;
}
