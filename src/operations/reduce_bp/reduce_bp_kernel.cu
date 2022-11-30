//
// Created by Luecx on 20.06.2022.
//
#include "reduce_bp.h"


__global__ void reduce_bp_kernel(
    const float* __restrict__ inp,
          float* __restrict__ inp_grd,
    const float* __restrict__ wgt,
          float* __restrict__ wgt_grd,
    const float* __restrict__ out_grd,
    unsigned int inp_size,
    unsigned int out_size,
    unsigned int bat_size,
    unsigned int reduction){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= bat_size || idy >= out_size)
        return;

    const float out_gradient = out_grd[MATRIX_INDEX(out_size, idy, idx)];

    const int offset_wgt = reduction * idy;
    const int offset_inp = offset_wgt + idx * inp_size;

    for(int i = 0; i < reduction; i++){
        inp_grd[offset_inp + i] = out_gradient * wgt[offset_wgt + i];
        atomicAdd(&wgt_grd[offset_wgt + i], out_gradient * inp[offset_inp + i]);
    }

}



