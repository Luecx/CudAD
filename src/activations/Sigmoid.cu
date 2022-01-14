
//
// Created by Luecx on 10.11.2021.
//

#include "Sigmoid.h"
#include "../misc/logging.h"
#include <assert.h>
#include <cmath>


__global__ void sigmoid_gpu(const float* __restrict__ vec,
                                  float* __restrict__ res,
                            const int size,
                            const float scalar){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= size) return;

    res[id] = 1.0f / (1 + expf(vec[id] * scalar));
}

__global__ void sigmoid_backprop_gpu(      float* __restrict__ vec_grd,
                                     const float* __restrict__ res,
                                     const float* __restrict__ res_grd,
                                     const int size,
                                     const float scalar){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= size) return;

    vec_grd[id] = scalar * res[id] * (1 - res[id]) * res_grd[id];
}


void Sigmoid::apply(DenseMatrix& inp, DenseMatrix& out, Mode mode) {

    assert(inp.values.size == out.values.size);

    if(mode == MODE_GPU){
        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)inp.values.size / block_size));
        sigmoid_gpu<<<grid, block>>>(
            inp.values.gpu_values,
            out.values.gpu_values,
            inp.values.size,
            scalar);
    }
}
void Sigmoid::backprop(DenseMatrix& inp, DenseMatrix& out, Mode mode) {

    assert(inp.values.size == out.values.size);

    if(mode == MODE_GPU){
        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)inp.values.size / block_size));
        sigmoid_backprop_gpu<<<grid, block>>>(
            inp.gradients.gpu_values,
            out.values   .gpu_values,
            out.gradients.gpu_values,
            inp.values.size,
            scalar);
    }

}

//void Sigmoid::apply(Data* inp, Data* out) {
//    for (int i = 0; i < out->M; i++) {
//        (*out)(i) = 1.0 / (1 + expf(-(*inp)(i) *SIGMOID_SCALE));
//    }
//}
//void Sigmoid::backprop(Data* out, Data* in_grad, Data* out_grad) {
//    for (int i = 0; i < out->M; i++) {
//        (*in_grad)(i) = (*out_grad)(i) * ((*out)(i) * (1 - (*out)(i))) * SIGMOID_SCALE;
//    }
//}
void Sigmoid::logOverview() { logging::write("Sigmoid (" + std::to_string(scalar) + ")"); }

#include "Sigmoid.h"
