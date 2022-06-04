
//
// Created by Luecx on 13.01.2022.
//

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