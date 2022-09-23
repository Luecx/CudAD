/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUDAD_SRC_OPERATIONS_MLE_MLE_H_
#define CUDAD_SRC_OPERATIONS_MLE_MLE_H_

#include "../../data/SArray.h"
#include "../../data/Mode.h"
#include "../../misc/config.h"

// clang-format off
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

    ASSERT(output.size() % 2 == 0);
    ASSERT(target.size() == output.size() / 2);
    ASSERT(target.size() == mask.size());
    ASSERT(loss.size()   >= 2);

    if(mode == DEVICE){

        ASSERT(output.gpuAddress());
        ASSERT(output_gradient.gpuAddress());
        ASSERT(target.gpuAddress());
        ASSERT(mask.gpuAddress());
        ASSERT(loss.gpuAddress());

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)output.size() / block_size));
        mle_kernel<<<grid, block>>>(
            output          .gpuAddress(),
            output_gradient .gpuAddress(),
            target          .gpuAddress(),
            mask            .gpuAddress(),
            loss            .gpuAddress(),
            target          .size());

    }else{
        ASSERT(false);
    }
}
// clang-format on

#endif    // CUDAD_SRC_OPERATIONS_MLE_MLE_H_
