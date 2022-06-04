//
// Created by Luecx on 30.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_BUCKET_BP_BUCKET_BP_H_
#define CUDAD_SRC_OPERATIONS_BUCKET_BP_BUCKET_BP_H_

#include "../../data/SArray.h"
#include "../../data/mode.h"

// clang-format off
__global__ void bucket_bp_kernel(
    const float* __restrict__ inp,
          float* __restrict__ inp_grd,
    const float* __restrict__ out_grd,
          float max_lower_bucket,
          float min_upper_bucket,
          int buckets,
          int input_size);

template<Mode mode>
inline void bucket_bp(const SArray<float> &inp,
                            SArray<float> &inp_grd,
                      const SArray<float> &out_grd,
                      float max_lower_bucket,
                      float min_upper_bucket){

    int bucket_size = out_grd.size / inp_grd.size;

    if(mode == DEVICE){

        ASSERT(inp.gpu_values);
        ASSERT(inp_grd.gpu_values);
        ASSERT(out_grd.gpu_values);

        constexpr int block_size_x = 1024;
        dim3 block(block_size_x);
        dim3 grid (std::ceil((float)inp.size / block_size_x));
        bucket_bp_kernel<<<grid, block>>>(
            inp.gpu_values,
            inp_grd.gpu_values,
            out_grd.gpu_values,
            max_lower_bucket,
            min_upper_bucket,
            bucket_size,
            inp.size);
    }else{
//        ASSERT(mat_grd.cpu_values);
//        ASSERT(vec_grd.cpu_values);
//        ASSERT(res_grd.cpu_values);
//
//        add_mv_bp_host(
//            mat_grd.cpu_values,
//            vec_grd.cpu_values,
//            res_grd.cpu_values,
//            mat_grd.m,
//            mat_grd.n,
//            mat_grd.leading_dimension,
//            res_grd.leading_dimension);
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_BUCKET_BP_BUCKET_BP_H_
