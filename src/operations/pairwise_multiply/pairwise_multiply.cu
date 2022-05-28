
//
// Created by Luecx on 28.05.2022.
//
#include "pairwise_multiply.h"

__global__ void pairwise_multiply_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
    unsigned int outsize){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= outsize) return;

    int idx1 = idx * 2;
    int idx2 = idx1 + 1;

    output[idx] = input[idx1] * input[idx2];

}