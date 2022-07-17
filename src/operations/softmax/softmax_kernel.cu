//
// Created by Luecx on 02.07.2022.
//
#include "softmax.h"

__global__ void softmax_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int m,
    unsigned int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n){
        return;
    }

    float division = 0;

    int offset = MATRIX_INDEX(m, 0, idx);

    // input regularization
    float mmax = A[offset];
    for (int i = 1; i < m; i++){
        mmax = max(mmax, A[offset + i]);
    }

    // denominator (adjust using input regularization)
    for (int i = 0; i < m; i++){
        division += exp(A[offset + i] - mmax);
    }

    // numerator (adjust using input regularization)
    for (int i = 0; i < m; i++){
        B[offset + i] = exp(A[offset + i] - mmax) / division;
    }

}
