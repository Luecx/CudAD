#include "../../data/mode.h"
#include "../../data/DenseMatrix.h"


#ifndef CUDATEST1_SRC_OPERATIONS_RADAM_RADAM_H_
#define CUDATEST1_SRC_OPERATIONS_RADAM_RADAM_H_

__global__ void radam_kernel(
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
          int N_sma_threshold);

void radam_host(
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
          int N_sma_threshold);

template<Mode mode>
inline void radam(SArray<float>& values,
                  SArray<float>& gradients,
                  SArray<float>& first_moment,
                  SArray<float>& second_moment,
                  int step,
                  float lr,
                  float beta1,
                  float beta2,
                  float eps,
                  int N_sma_threshold) {
    constexpr int block_size = 1024;

    ASSERT(values.size == gradients.size)
    ASSERT(values.size == first_moment.size)
    ASSERT(values.size == second_moment.size)

    if(mode == DEVICE) {
        dim3 block(block_size);
        dim3 grid (std::ceil((float)values.size / block_size));
        radam_kernel<<<grid, block>>>(
            values          .gpu_values,
            gradients       .gpu_values,
            first_moment    .gpu_values,
            second_moment   .gpu_values,
            values.size,
            step, lr, beta1, beta2, eps, N_sma_threshold);
    } else {
        radam_host(
            values          .cpu_values,
            gradients       .cpu_values,
            first_moment    .cpu_values,
            second_moment   .cpu_values,
            values.size,
            step, lr, beta1, beta2, eps, N_sma_threshold);
    }
}


#endif //CUDATEST1_SRC_OPERATIONS_RADAM_RADAM_H_
