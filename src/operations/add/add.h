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

#ifndef CUDATEST1_SRC_OPERATIONS_ADD_ADD_H_
#define CUDATEST1_SRC_OPERATIONS_ADD_ADD_H_

#include "../../data/SArray.h"
#include "../../data/Mode.h"

#include <iostream>

// clang-format off
void add_host(
    const float* A,
    const float* B,
          float* C,
          unsigned int A_size,
          unsigned int B_size,
          unsigned int C_size,
          unsigned int size,
          float alpha,
          float beta);

__global__ void add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
          unsigned int A_size,
          unsigned int B_size,
          unsigned int C_size,
          unsigned int size,
          float alpha,
          float beta);
// clang-format on

/**
 * performs C = A * alpha + B * beta
 * If A and B are not the same size as C, it will repeat the data contained in A and B
 * @tparam mode
 * @param A
 * @param B
 * @param C
 * @param alpha
 * @param beta
 */
template<Mode mode>
// clang-format off
inline void add   ( const SArray<float> &A,
                    const SArray<float> &B,
                          SArray<float> &C,
                          float alpha,
                          float beta,
                          int size = -1){


    if (size == -1){
        size = C.size();
    }

    if(mode == DEVICE){

        ASSERT(A.gpuAddress());
        ASSERT(B.gpuAddress());
        ASSERT(C.gpuAddress());

        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)size / block_size));
        add_kernel<<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            C.gpuAddress(),
            A.size(),
            B.size(),
            C.size(),
            size,
            alpha,
            beta);
    }else{
        add_host(
            A.cpuAddress(),
            B.cpuAddress(),
            C.cpuAddress(),
            A.size(),
            B.size(),
            C.size(),
            size,
            alpha,
            beta);
    }
}
// clang-format on

#endif    // CUDATEST1_SRC_OPERATIONS_ADD_ADD_H_
