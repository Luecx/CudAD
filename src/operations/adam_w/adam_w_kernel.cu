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

// https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py#L96

// clang-format off
__global__ void adam_w_kernel(
          float* __restrict__ values,
          float* __restrict__ gradients,
          float* __restrict__ exp_avg,
          float* __restrict__ exp_avg_sq,
          int   size,
          int   step,
          float lr,
          float beta1,
          float beta2,
          float eps,
          int   warmup) {
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    // clang-format off
    exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1.0 - beta2) * gradients[idx] * gradients[idx];
    exp_avg   [idx] = beta1 * exp_avg   [idx] + (1.0 - beta1) * gradients[idx];
    // clang-format on

    // we increment step in the struct, no need to do it here

    float denom        = sqrtf(exp_avg_sq[idx]) + eps;
    float bc1          = 1.0 - powf(beta1, step);
    float bc2          = 1.0 - powf(beta2, step);

    float scheduled_lr = lr;
    if (warmup > step)
        scheduled_lr = 1e-8 + step * lr / warmup;

    float step_size = scheduled_lr * sqrtf(bc2) / bc1;
    float delta     = step_size * exp_avg[idx] / denom;

    values[idx] -= delta;
    gradients[idx] = 0;
}