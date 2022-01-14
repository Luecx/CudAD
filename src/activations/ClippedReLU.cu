


#include <assert.h>
#include "ClippedReLU.h"
#include "../misc/logging.h"

//void ClippedReLU::apply(Data* inp, Data* out) {
//
//    assert(inp->M == out->M);
//
//    __m256    lower      = _mm256_set1_ps(0);
//    __m256    upper      = _mm256_set1_ps(max);
//
//    float*    outputVals = out->values;
//    float*    inputVals  = inp->values;
//
//    const int size       = PARALLEL_SIZE_32_BIT(inp->M);
//
//    for (int i = 0; i < size; i += 8) {
//        __m256 in  = _mm256_load_ps(&(inputVals[i]));
//        __m256 out = _mm256_min_ps(upper, _mm256_max_ps(in, lower));
//
//        _mm256_store_ps(&(outputVals[i]), out);
//    }
//    for (int i = size; i < inp->M; i++) {
//        outputVals[i] = inputVals[i] < 0 ? 0 : inputVals[i];
//    }
//}
//void ClippedReLU::backprop(Data* out, Data* in_grad, Data* out_grad) {
//    assert(out->M == out->M);
//
//    static __m256    lower      = _mm256_set1_ps(0);
//    static __m256    upper      = _mm256_set1_ps(max);
//    static const int operaL      = 30;    // _CMP_GT_OQ
//    static const int operaU      = 17;    // _CMP_LT_OQ
//
//    auto*          outputVals = (__m256*) out->values;
//    auto*          inp_grad   = (__m256*) in_grad->values;
//    auto*          oup_grad   = (__m256*) out_grad->values;
//
//    const int        size       = PARALLEL_SIZE_32_BIT(out->M);
//
//    for (int i = 0; i < size / 8; i++) {
//        auto maskLower = _mm256_cmp_ps(outputVals[i], lower, operaL); // mask all bits larger than 0
//        auto maskUpper = _mm256_cmp_ps(outputVals[i], upper, operaU); // mask all bits lower than max
//        inp_grad[i] = _mm256_blendv_ps(lower, oup_grad[i], maskLower);  // blend all which are larger than 0
//        inp_grad[i] = _mm256_blendv_ps(lower, inp_grad[i], maskUpper);  // blend all which are lower than max
//    }
//
//    for (int i = size; i < out->M; i++) {
//        (*in_grad)(i) = ((*out)(i) > 0 && (*out)(i) < max) ? (*out_grad)(i) : 0;
//    }
//}


__global__ void clipped_relu_gpu(const float* __restrict__ vec,
                                       float* __restrict__ res,
                                 const int size,
                                 const float top_margin){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= size) return;

    if     (vec[id] < 0)            res[id] = 0;
    else if(vec[id] > top_margin)   res[id] = top_margin;
    else                            res[id] = vec[id];
}

__global__ void clipped_relu_backprop_gpu(      float* __restrict__ vec_grd,
                                          const float* __restrict__ res,
                                          const float* __restrict__ res_grd,
                                          const int size,
                                          const float top_margin){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= size) return;

    if     (res[id] < 0)            vec_grd[id] = 0;
    else if(res[id] > top_margin)   vec_grd[id] = 0;
    else                            vec_grd[id] = res_grd[id];

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

void ClippedReLU::apply(DenseMatrix& inp, DenseMatrix& out, Mode mode) {

    assert(inp.values.size == out.values.size);

    if(mode == MODE_GPU){
        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)inp.values.size / block_size));
        clipped_relu_gpu<<<grid, block>>>(
            inp.values.gpu_values,
            out.values.gpu_values,
            inp.values.size,
            max);
    }
}
void ClippedReLU::backprop(DenseMatrix& inp, DenseMatrix& out, Mode mode) {

    assert(inp.values.size == out.values.size);

    if(mode == MODE_GPU){
        constexpr int block_size = 1024;
        dim3 block(block_size);
        dim3 grid (std::ceil((float)inp.values.size / block_size));
        clipped_relu_backprop_gpu<<<grid, block>>>(
            inp.gradients.gpu_values,
            out.values   .gpu_values,
            out.gradients.gpu_values,
            inp.values.size,
            max);
    }

}

void ClippedReLU::logOverview() {logging::write("ClippedReLU (" + std::to_string(max) + ")");}

#include "ClippedReLU.h"
