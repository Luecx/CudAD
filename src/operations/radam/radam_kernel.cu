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
__global__ void radam_kernel(
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
          int   N_sma_threshold) {

    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1.0 - beta2) * gradients[idx] * gradients[idx];
    exp_avg[idx]    = beta1 * exp_avg[idx] + (1.0 - beta1) * gradients[idx];

    // we increment step in the struct, no need to do it here

    float beta2_t   = powf(beta2, step);
    float N_sma_max = 2.0 / (1.0 - beta2) - 1.0;
    float N_sma     = N_sma_max - 2 * step * beta2_t / (1.0 - beta2_t);

    if (N_sma >= N_sma_threshold) {
        float step_size = lr
                          * sqrtf((1.0 - beta2_t) * (N_sma - 4.0) / (N_sma_max - 4.0) * (N_sma - 2.0)
                                  / N_sma * N_sma_max / (N_sma_max - 2.0))
                          / (1.0 - powf(beta1, step));

        float denom = sqrtf(exp_avg_sq[idx]) + eps;
        float delta = step_size * exp_avg[idx] / denom;

        values[idx] -= delta;
    } else {
        float step_size = lr * (1.0 - powf(beta1, step));
        float delta     = step_size * exp_avg[idx];

        values[idx] -= delta;
    }

    gradients[idx] = 0;
}