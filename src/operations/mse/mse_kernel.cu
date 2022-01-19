
//
// Created by Luecx on 18.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MSE_MSE_KERNEL_CU_
#define CUDAD_SRC_OPERATIONS_MSE_MSE_KERNEL_CU_


__global__ void mse_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
          float* __restrict__ loss,
    unsigned int size){


    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size) return;

    float difference = output[idx] - target[idx];

    output_gradient[idx] = 2 * difference / size;

    atomicAdd(loss, difference * difference / size);
}
#endif    // CUDAD_SRC_OPERATIONS_MSE_MSE_KERNEL_CU_
