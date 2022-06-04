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

#ifndef CUDAD_SRC_OPERATIONS_MLE_MLE_KERNEL_CU_
#define CUDAD_SRC_OPERATIONS_MLE_MLE_KERNEL_CU_

// clang-format off
__global__ void mle_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
    const bool * __restrict__ mask,
          float* __restrict__ loss,
    unsigned int size){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    // this loss uses a single target value and trains consecutive values of mean and variance
    // therefor each thread uses one target and sets the gradients of two outputs
    int idx_mean = idx * 2;
    int idx_vari = idx_mean + 1;

    if (mask[idx]) {
        float difference          = output[idx_mean] - target[idx];
        float variance            = output[idx_vari];

        float current_loss        = 0.5 * (difference * difference / variance + logf(variance));

        output_gradient[idx_mean] = difference / (size * variance);
        output_gradient[idx_vari] =
            (variance - difference * difference) / (2 * size * variance * variance);

        //        output_gradient[idx] = 2 * difference / size;
        atomicAdd(&loss[0], current_loss / size);
        atomicAdd(&loss[1], difference * difference / size);
    } else {
        output_gradient[idx] = 0;
    }
}
#endif    // CUDAD_SRC_OPERATIONS_MLE_MLE_KERNEL_CU_
