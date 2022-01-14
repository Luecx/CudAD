

//
// Created by Luecx on 11.01.2022.
//
#include "../../data/mode.h"
#include "../../data/DenseMatrix.h"


#ifndef CUDATEST1_SRC_OPERATIONS_ADAM_ADAM_H_
#define CUDATEST1_SRC_OPERATIONS_ADAM_ADAM_H_

__global__ void adam_kernel(
          float* __restrict__ values,
    const float* __restrict__ gradients,
          float* __restrict__ first_moment,
          float* __restrict__ second_moment,
          int size,
          float alpha,
          float beta1,
          float beta2,
          float eps);

void adam_host(
          float* values,
    const float* gradients,
          float* first_moment,
          float* second_moment,
          int size,
          float alpha,
          float beta1,
          float beta2,
          float eps);

template<Mode mode>
inline void adam(SArray<float>& values,
                 SArray<float>& gradients,
                 SArray<float>& first_moment,
                 SArray<float>& second_moment,
                 float alpha,
                 float beta1,
                 float beta2,
                 float eps){

    constexpr int block_size = 1024;

    ASSERT(values.size == gradients.size)
    ASSERT(values.size == first_moment.size)
    ASSERT(values.size == second_moment.size)

    if(mode == DEVICE){
        dim3 block(block_size);
        dim3 grid (std::ceil((float)values.size / block_size));
        adam_kernel<<<grid, block>>>(
            values          .gpu_values,
            gradients       .gpu_values,
            first_moment    .gpu_values,
            second_moment   .gpu_values,
            values.size,
            alpha, beta1, beta2, eps);
    }else{
        adam_host(
            values          .cpu_values,
            gradients       .cpu_values,
            first_moment    .cpu_values,
            second_moment   .cpu_values,
            values.size,
            alpha, beta1, beta2, eps);
    }
    cudaDeviceSynchronize();
}


#endif //CUDATEST1_SRC_OPERATIONS_ADAM_ADAM_H_
