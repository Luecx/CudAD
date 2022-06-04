
//
// Created by Luecx on 30.01.2022.
//
#include "bucket_bp.h"

// clang-format off
__global__ void bucket_bp_kernel(
    const float* __restrict__ inp,
          float* __restrict__ inp_grd,
    const float* __restrict__ out_grd,
          float max_lower_bucket,
          float min_upper_bucket,
          int buckets,
          int input_size){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_size)
        return;

    float bucket_size      = (min_upper_bucket - max_lower_bucket) / (buckets - 2);

    int   inner_bucket_idx = ceil((inp[idx] - max_lower_bucket) / bucket_size);

    if (inner_bucket_idx < 0)
        inner_bucket_idx = 0;
    if (inner_bucket_idx >= buckets)
        inner_bucket_idx = buckets - 1;

    // clear the output
    int outp_offset = buckets * idx;

    inp_grd[idx]    = out_grd[outp_offset + inner_bucket_idx];
}