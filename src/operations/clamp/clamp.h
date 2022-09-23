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

#include "../../data/DenseMatrix.h"
#include "../../data/Mode.h"

#ifndef CUDATEST1_SRC_OPERATIONS_CLAMP_CLAMP_H_
#define CUDATEST1_SRC_OPERATIONS_CLAMP_CLAMP_H_

// clang-format off
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
        dim3 grid (std::ceil((float)values.size() / block_size));
        clamp_kernel<<<grid, block>>>(
            values.gpuAddress(),min,max,values.size());
    }else{
        clamp_host(
            values.cpuAddress(),min,max,values.size());
    }
//    cudaDeviceSynchronize();
}

// clang-format on

#endif    // CUDATEST1_SRC_OPERATIONS_CLAMP_CLAMP_H_
