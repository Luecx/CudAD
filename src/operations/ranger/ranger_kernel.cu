// https://github.com/lessw2020/Best-Deep-Learning-Optimizers/blob/master/Ranger/ranger.py

__global__ void ranger_kernel(
          float* __restrict__ values,
          float* __restrict__ gradients,
          float* __restrict__ exp_avg,
          float* __restrict__ exp_avg_sq,
          float* __restrict__ slow_buffer,
          int   size,
          int   step,
          float lr,
          float beta1,
          float beta2,
          float eps,
          float alpha,
          int   k,
          int   N_sma_threshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;

    exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1.0 - beta2) * gradients[idx] * gradients[idx];
    exp_avg   [idx] = beta1 * exp_avg   [idx] + (1.0 - beta1) * gradients[idx];

    // we increment step in the struct, no need to do it here
    
    float beta2_t = powf(beta2, step);
    float N_sma_max = 2.0 / (1.0 - beta2) - 1.0;
    float N_sma = N_sma_max - 2 * step * beta2_t / (1.0 - beta2_t);

    if (N_sma >= N_sma_threshold) {
        float step_size = lr * sqrtf(
            (1.0 - beta2_t) * 
            (N_sma - 4.0) / (N_sma_max - 4.0) * 
            (N_sma - 2.0) / N_sma * 
            N_sma_max / (N_sma_max - 2.0)
        ) / (1.0 - powf(beta1, step));

        float denom = sqrtf(exp_avg_sq[idx]) + eps;
        float delta = step_size * exp_avg[idx] / denom;

        values[idx] -= delta;
    } else {
        float step_size = lr * (1.0 - powf(beta1, step));
        float delta = step_size * exp_avg[idx];

        values[idx] -= delta;
    }

    if (step % k == 0) {
        slow_buffer[idx] += alpha * (values[idx] - slow_buffer[idx]);
        values[idx] = slow_buffer[idx];
    }

    gradients[idx] = 0;
}