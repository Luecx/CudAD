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

#ifndef CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_PAIRWISE_MULTIPLY_H_
#define CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_PAIRWISE_MULTIPLY_H_

#include "../../data/SArray.h"
#include "../../data/Mode.h"
#include "../../misc/config.h"

// clang-format off
__global__ void pairwise_multiply_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
    unsigned int outsize);

template<Mode mode>
inline void pairwise_multiply (
               const  SArray<float>& input,
                      SArray<float>& output){
    if(mode == DEVICE){

        ASSERT(input.gpuAddress());
        ASSERT(output.gpuAddress());
        ASSERT(input.size() == 2 * output.size());

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)output.size() / block_size));
        pairwise_multiply_kernel<<<grid, block>>>(
            input .gpuAddress(),
            output.gpuAddress(),
            output.size());
    }else{
        ASSERT(false);
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_PAIRWISE_MULTIPLY_PAIRWISE_MULTIPLY_H_
