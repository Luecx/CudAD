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

#ifndef CUDATEST1_SRC_OPERATIONS_SIGMOID_SIGMOID_H_
#define CUDATEST1_SRC_OPERATIONS_SIGMOID_SIGMOID_H_

#include "../../data/SArray.h"
#include "../../data/Mode.h"

#include <iostream>

// clang-format off
void sigmoid_host(
    const float* A,
          float* B,
    unsigned int size,
    float scalar);

__global__ void sigmoid_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size,
    float scalar);

template<Mode mode>
inline void sigmoid   (const SArray<float> &A,
                             SArray<float> &B,
                             float scalar){

    ASSERT(A.size() == B.size())

    if(mode == DEVICE){

        ASSERT(A.gpuAddress());
        ASSERT(B.gpuAddress());

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)A.size() / block_size));
        sigmoid_kernel<<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            A.size(),
            scalar);
    }else{
        sigmoid_host(
            A.cpuAddress(),
            B.cpuAddress(),
            A.size(),
            scalar);
    }
}
// clang-format on

#endif    // CUDATEST1_SRC_OPERATIONS_SIGMOID_SIGMOID_H_
