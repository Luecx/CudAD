//
// Created by Luecx on 28.05.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_PAIRWISE_MULTIPLY_H_
#define CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_PAIRWISE_MULTIPLY_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include "../../misc/config.h"


__global__ void pairwise_multiply_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
    unsigned int outsize);

template<Mode mode>
inline void pairwise_multiply (
               const  SArray<float>& input,
                      SArray<float>& output){
    if(mode == DEVICE){

        ASSERT(input.gpu_values);
        ASSERT(output.gpu_values);
        ASSERT(input.size == 2 * output.size);

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)output.size / block_size));
        pairwise_multiply_kernel<<<grid, block>>>(
            input .gpu_values,
            output.gpu_values,
            output.size);
    }else{
        ASSERT(false);
    }
}

#endif    // CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_PAIRWISE_MULTIPLY_H_
