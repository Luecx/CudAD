

//
// Created by Luecx on 18.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MLE_MLE_H_
#define CUDAD_SRC_OPERATIONS_MLE_MLE_H_

#include "../../data/mode.h"
#include "../../data/SArray.h"
#include "../../misc/config.h"

__global__ void mle_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
    const bool * __restrict__ mask,
          float* __restrict__ loss,
    unsigned int size);



template<Mode mode>
inline void mle (const SArray<float>& output,
                       SArray<float>& output_gradient,
                 const SArray<float>& target,
                 const SArray< bool>& mask,
                       SArray<float>& loss){

    ASSERT(output.size % 2 == 0);
    ASSERT(target.size == output.size / 2);
    ASSERT(target.size == mask.size);
    ASSERT(loss.size   >= 2);

    if(mode == DEVICE){

        ASSERT(output.gpu_values);
        ASSERT(output_gradient.gpu_values);
        ASSERT(target.gpu_values);
        ASSERT(mask.gpu_values);
        ASSERT(loss.gpu_values);

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)output.size / block_size));
        mle_kernel<<<grid, block>>>(
            output          .gpu_values,
            output_gradient .gpu_values,
            target          .gpu_values,
            mask            .gpu_values,
            loss            .gpu_values,
            target.size);

    }else{
        ASSERT(false);
    }
}


#endif    // CUDAD_SRC_OPERATIONS_MLE_MLE_H_
