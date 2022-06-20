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

#ifndef CUDAD_SRC_OPERATIONS_BUCKET_BP_BUCKET_BP_H_
#define CUDAD_SRC_OPERATIONS_BUCKET_BP_BUCKET_BP_H_

#include "../../data/SArray.h"
#include "../../data/Mode.h"

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

    int bucket_size = out_grd.size() / inp_grd.size();

    if(mode == DEVICE){

        ASSERT(inp.gpu_address());
        ASSERT(inp_grd.gpu_address());
        ASSERT(out_grd.gpu_address());

        constexpr int block_size_x = 1024;
        dim3 block(block_size_x);
        dim3 grid (std::ceil((float)inp.size() / block_size_x));
        bucket_bp_kernel<<<grid, block>>>(
            inp.gpu_address(),
            inp_grd.gpu_address(),
            out_grd.gpu_address(),
            max_lower_bucket,
            min_upper_bucket,
            bucket_size,
            inp.size());
    }else{
//        ASSERT(mat_grd.cpu_address());
//        ASSERT(vec_grd.cpu_address());
//        ASSERT(res_grd.cpu_address());
//
//        add_mv_bp_host(
//            mat_grd.cpu_address(),
//            vec_grd.cpu_address(),
//            res_grd.cpu_address(),
//            mat_grd.m,
//            mat_grd.n,
//            mat_grd.leading_dimension,
//            res_grd.leading_dimension);
    }
}

// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_BUCKET_BP_BUCKET_BP_H_
