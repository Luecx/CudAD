
//
// Created by Luecx on 13.01.2022.
//

__global__ void clamp_kernel(
    float* __restrict__ values,
    float p_min,
    float p_max,
    float size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;

    values[idx] = max(p_min, min(p_max, values[idx]));
}