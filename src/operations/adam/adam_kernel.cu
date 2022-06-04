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

// clang-format off
__global__ void adam_kernel(
          float* __restrict__ values,
          float* __restrict__ gradients,
          float* __restrict__ first_moment,
          float* __restrict__ second_moment,
          int   size,
          float alpha,
          float beta1,
          float beta2,
          float eps){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    first_moment[idx]  = beta1 * first_moment[idx] + (1 - beta1) * gradients[idx];
    second_moment[idx] = beta2 * second_moment[idx] + (1 - beta2) * gradients[idx] * gradients[idx];

    float delta        = alpha * first_moment[idx] / (sqrtf(second_moment[idx]) + eps);
    values[idx] -= delta;
    gradients[idx] = 0;
}