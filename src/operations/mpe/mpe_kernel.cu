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

#ifndef CUDAD_SRC_OPERATIONS_MPE_MPE_KERNEL_CU_
#define CUDAD_SRC_OPERATIONS_MPE_MPE_KERNEL_CU_

#include <stdio.h>

// clang-format off
__global__ void mpe_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
    const bool * __restrict__ mask,
          float* __restrict__ loss,
          float power,
    unsigned int size,
    unsigned int grad_division){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    if (mask[idx]) {
        float difference     = output[idx] - target[idx];
        float abs_diff       = abs(difference);
        float sign           = difference > 0 ? 1 : -1;

        float derivative     = powf(abs_diff, power - 1) * sign * power;
        float loss_val       = powf(abs_diff, power);

        output_gradient[idx] = derivative;
        atomicAdd(loss, loss_val / size);
    } else {
        output_gradient[idx] = 0;
    }
}
#endif    // CUDAD_SRC_OPERATIONS_MPE_MPE_KERNEL_CU_
