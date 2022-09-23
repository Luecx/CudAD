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

#ifndef CUDATEST1_SRC_OPERATIONS_ADAM_W_ADAM_W_H_
#define CUDATEST1_SRC_OPERATIONS_ADAM_W_ADAM_W_H_

// clang-format off
__global__ void adam_w_kernel(
          float* __restrict__ values,
          float* __restrict__ gradients,
          float* __restrict__ exp_avg,
          float* __restrict__ exp_avg_sq,
          int size,
          int step,
          float lr,
          float beta1,
          float beta2,
          float eps,
          int warmup);

void adam_w_host(
          float* values,
          float* gradients,
          float* exp_avg,
          float* exp_avg_sq,
          int size,
          int step,
          float lr,
          float beta1,
          float beta2,
          float eps,
          int warmup);

template<Mode mode>
inline void adam_w(SArray<float>& values,
                   SArray<float>& gradients,
                   SArray<float>& first_moment,
                   SArray<float>& second_moment,
                   int step,
                   float lr,
                   float beta1,
                   float beta2,
                   float eps,
                   int warmup) {
    constexpr int block_size = 1024;

    ASSERT(values.size == gradients.size)
    ASSERT(values.size == first_moment.size)
    ASSERT(values.size == second_moment.size)

    if(mode == DEVICE) {
        dim3 block(block_size);
        dim3 grid (std::ceil((float)values.size / block_size));
        adam_w_kernel<<<grid, block>>>(
            values          .gpuAddress(),
            gradients       .gpuAddress(),
            first_moment    .gpuAddress(),
            second_moment   .gpuAddress(),
            values.size,
            step, lr, beta1, beta2, eps, warmup);
    } else {
        adam_w_host(
            values          .cpuAddress(),
            gradients       .cpuAddress(),
            first_moment    .cpuAddress(),
            second_moment   .cpuAddress(),
            values.size,
            step, lr, beta1, beta2, eps, warmup);
    }
}
// clang-format on

#endif    // CUDATEST1_SRC_OPERATIONS_ADAM_W_ADAM_W_H_
