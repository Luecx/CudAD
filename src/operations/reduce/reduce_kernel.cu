//
// Created by Luecx on 20.06.2022.
//
#include "reduce.h"

__global__ void reduce_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ wgt,
          float* __restrict__ out,
    unsigned int inp_size,
    unsigned int out_size,
    unsigned int bat_size,
    unsigned int reduction){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= bat_size || idy >= out_size)
        return;

    float sum = 0;

    const int offset_wgt = reduction * idy;
    const int offset_inp = offset_wgt + idx * inp_size;

    for(int i = 0; i < reduction; i++){
        sum += inp[offset_inp + i]* wgt[offset_wgt + i];
    }

    out[MATRIX_INDEX(out_size, idy, idx)] = sum;
}

