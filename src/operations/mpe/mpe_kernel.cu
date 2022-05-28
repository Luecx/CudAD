
//
// Created by Luecx on 18.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MPE_MPE_KERNEL_CU_
#define CUDAD_SRC_OPERATIONS_MPE_MPE_KERNEL_CU_

#include <stdio.h>

__global__ void mpe_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
    const bool * __restrict__ mask,
          float* __restrict__ loss,
          float power,
    unsigned int size,
    unsigned int grad_division){


    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size) return;

    if(mask[idx]){
        float difference = output[idx] - target[idx];
        float abs_diff   = abs(difference);
        float sign       = difference > 0 ? 1 : -1;

        float derivative = powf(abs_diff, power-1) * sign;
        float loss_val   = powf(abs_diff, power);

        output_gradient[idx] = derivative;
        atomicAdd(loss, loss_val / size);
    }else{
        output_gradient[idx] = 0;
    }
}
#endif    // CUDAD_SRC_OPERATIONS_MPE_MPE_KERNEL_CU_
