

//
// Created by Luecx on 18.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MSE_MSE_H_
#define CUDAD_SRC_OPERATIONS_MSE_MSE_H_

#include "../../data/SArray.h"
#include "../../data/mode.h"
#include "../../misc/config.h"

// clang-format off
__global__ void mse_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
    const bool * __restrict__ mask,
          float* __restrict__ loss,
    unsigned int size);



template<Mode mode>
inline void mse (const SArray<float>& output,
                       SArray<float>& output_gradient,
                 const SArray<float>& target,
                 const SArray< bool>& mask,
                       SArray<float>& loss){

    if(mode == DEVICE){

        ASSERT(output.gpu_values);
        ASSERT(output_gradient.gpu_values);
        ASSERT(target.gpu_values);
        ASSERT(mask.gpu_values);
        ASSERT(loss.gpu_values);


        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)output.size / block_size));
        mse_kernel<<<grid, block>>>(
            output          .gpu_values,
            output_gradient .gpu_values,
            target          .gpu_values,
            mask            .gpu_values,
            loss            .gpu_values,
            output.size);

    }else{
        ASSERT(false);
    }
}

// clang-format on

#endif    // CUDAD_SRC_OPERATIONS_MSE_MSE_H_
