//
// Created by Luecx on 30.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_BUCKET_BUCKET_H_
#define CUDAD_SRC_OPERATIONS_BUCKET_BUCKET_H_

#include "../../data/SArray.h"
#include "../../data/mode.h"

// clang-format off
__global__ void bucket_kernel(
    const float* __restrict__ inp,
          float* __restrict__ out,
          float max_lower_bucket,
          float min_upper_bucket,
          int buckets,
          int input_size);

template<Mode mode>
inline void bucket  ( const SArray<float> &inp,
                      const SArray<float> &out,
                      float max_lower_bucket,
                      float min_upper_bucket){

    int bucket_size = out.size / inp.size;

    if(mode == DEVICE){

        ASSERT(inp.gpu_values);
        ASSERT(out.gpu_values);

        constexpr int block_size_x = 1024;
        dim3 block(block_size_x);
        dim3 grid (std::ceil((float)inp.size / block_size_x));
        bucket_kernel<<<grid, block>>>(
            inp.gpu_values,
            out.gpu_values,
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

#endif    // CUDAD_SRC_OPERATIONS_BUCKET_BUCKET_H_
