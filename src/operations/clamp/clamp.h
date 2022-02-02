

//
// Created by Luecx on 11.01.2022.
//
#include "../../data/mode.h"
#include "../../data/DenseMatrix.h"


#ifndef CUDATEST1_SRC_OPERATIONS_CLAMP_CLAMP_H_
#define CUDATEST1_SRC_OPERATIONS_CLAMP_CLAMP_H_

__global__ void clamp_kernel(
        float* __restrict__ values,
        float min,
        float max,
        float size);

void clamp_host(
            float* values,
            float min,
            float max,
            float size);

template<Mode mode>
inline void clamp(SArray<float>& values,
                  float min,
                  float max){



    if(mode == DEVICE){
        constexpr int block_size = 1024;

        dim3 block(block_size);
        dim3 grid (std::ceil((float)values.size / block_size));
        clamp_kernel<<<grid, block>>>(
            values.gpu_values,min,max,values.size);
    }else{
        clamp_host(
            values.cpu_values,min,max,values.size);
    }
//    cudaDeviceSynchronize();
}


#endif //CUDATEST1_SRC_OPERATIONS_CLAMP_CLAMP_H_
