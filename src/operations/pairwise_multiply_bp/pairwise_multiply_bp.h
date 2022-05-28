//
// Created by Luecx on 28.05.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_BP_PAIRWISE_MULTIPLY_BP_H_
#define CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_BP_PAIRWISE_MULTIPLY_BP_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include "../../misc/config.h"

__global__ void pairwise_multiply_bp_kernel(
    const float* __restrict__ input,
          float* __restrict__ input_grd,
    const float* __restrict__ output_grd,
    unsigned int outsize);

template<Mode mode>
inline void pairwise_multiply_bp (
               const SArray<float>& input,
                     SArray<float>& input_grd,
               const SArray<float>& output_grd){
    if(mode == DEVICE){

        ASSERT(input.gpu_values);
        ASSERT(output.gpu_values);
        ASSERT(input_grd.gpu_values);
        ASSERT(output_grd.gpu_values);
        ASSERT(input.size == 2 * output_grd.size);
        ASSERT(input.size == input_grd.size);

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)output_grd.size / block_size));
        pairwise_multiply_bp_kernel<<<grid, block>>>(
            input     .gpu_values,
            input_grd .gpu_values,
            output_grd.gpu_values,
            output_grd.size);
    }else{
        ASSERT(false);
    }
}

#endif    // CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_BP_PAIRWISE_MULTIPLY_BP_H_
