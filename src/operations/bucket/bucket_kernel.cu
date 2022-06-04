/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "bucket.h"

// clang-format off
__global__ void bucket_kernel(
    const float* __restrict__ inp,
          float* __restrict__ out,
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
    for (int i = 0; i < buckets; i++) {
        out[outp_offset + i] = 0;
    }

    // set the correct value
    out[outp_offset + inner_bucket_idx] = inp[idx];
}