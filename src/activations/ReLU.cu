
//
// Created by Luecx on 10.11.2021.
//

#include "ReLU.h"
#include <assert.h>
#include "../misc/logging.h"


__global__ void relu_gpu(const float* __restrict__ vec,
                               float* __restrict__ res,
                         const int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= size) return;

    if(vec[id] > 0) res[id] = vec[id];
    else res[id] = 0;
}

__global__ void relu_backprop_gpu(      float* __restrict__ vec_grd,
                                  const float* __restrict__ res,
                                  const float* __restrict__ res_grd,
                                  const int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= size) return;

    if(res[id] > 0) vec_grd[id] = res_grd[id];
    else vec_grd[id] = 0;
}


//void relu_cpu(float* inp, float* out){
//    assert(inp->M == out->M);
//
//    __m256    lower      = _mm256_set1_ps(0);
//
//    const int size       = PARALLEL_SIZE_32_BIT(inp->M);
//
//    for (int i = 0; i < size; i += 8) {
//        __m256 in  = _mm256_load_ps(&(inputVals[i]));
//        __m256 out = _mm256_max_ps(in, lower);
//
//        _mm256_store_ps(&(outputVals[i]), out);
//    }
//    for (int i = size; i < inp->M; i++) {
//        outputVals[i] = inputVals[i] < 0 ? 0 : inputVals[i];
//    }
//}


//void relu_gpu(float* inp, float* out){
//    assert(out->M == out->M);
//    static __m256    lower      = _mm256_set1_ps(0);
//    static const int opera      = 30;    // _CMP_GT_OQ
//
//    auto*          outputVals = (__m256*) out->values;
//    auto*          inp_grad   = (__m256*) in_grad->values;
//    auto*          oup_grad   = (__m256*) out_grad->values;
//
//    const int        size       = PARALLEL_SIZE_32_BIT(out->M);
//
//    for (int i = 0; i < size / 8; i++) {
//        auto mask = _mm256_cmp_ps(outputVals[i], lower, opera);
//        inp_grad[i] = _mm256_blendv_ps(lower, oup_grad[i], mask);
//    }
//
//    for (int i = size; i < out->M; i++) {
//        (*in_grad)(i) = (*out)(i) > 0 ? (*out_grad)(i) : 0;
//    }}

void ReLU::apply(DenseMatrix& inp, DenseMatrix& out, Mode mode) {

    assert(inp.values.size == out.values.size);

    if(mode == MODE_GPU){
        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)inp.values.size / block_size));
        relu_gpu<<<grid, block>>>(
            inp.values.gpu_values,
            out.values.gpu_values,
            inp.values.size);
    }
}
void ReLU::backprop(DenseMatrix& inp, DenseMatrix& out, Mode mode) {

    assert(inp.values.size == out.values.size);

    if(mode == MODE_GPU){
        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)inp.values.size / block_size));
        relu_backprop_gpu<<<grid, block>>>(
            inp.gradients.gpu_values,
            out.values   .gpu_values,
            out.gradients.gpu_values,
            inp.values.size);
    }

}
void ReLU::logOverview() {logging::write("ReLU");}
