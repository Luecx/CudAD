// https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py#L96

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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;

    exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1.0 - beta2) * gradients[idx] * gradients[idx];
    exp_avg   [idx] = beta1 * exp_avg   [idx] + (1.0 - beta1) * gradients[idx];

    // we increment step in the struct, no need to do it here

    float denom = sqrtf(exp_avg_sq[idx]) + eps;
    float bc1   = 1.0 - powf(beta1, step);
    float bc2   = 1.0 - powf(beta2, step);

    float scheduled_lr = lr;
    if (warmup > step)
        scheduled_lr = 1e-8 + step * lr / warmup;

    float step_size = scheduled_lr * sqrtf(bc2) / bc1;
    float delta     = step_size * exp_avg[idx] / denom;

    values[idx]   -= delta;
    gradients[idx] = 0;
}