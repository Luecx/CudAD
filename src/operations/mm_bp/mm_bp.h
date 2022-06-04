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

#ifndef CUDAD_SRC_OPERATIONS_MM_BP_MM_BP_H_
#define CUDAD_SRC_OPERATIONS_MM_BP_MM_BP_H_

#include "../mm/mm_cublas.h"

// clang-format off
template<Mode mode>
inline void mm_bp(DenseMatrix &mat1,
        DenseMatrix &mat1_grd,
        DenseMatrix &mat2,
        DenseMatrix &mat2_grd,
        DenseMatrix &res_grd){

    ASSERT(mat1_grd.n == mat2_grd.m);
    ASSERT(mat1_grd.m == res_grd .m);
    ASSERT(mat2_grd.n == res_grd .n);

    if(mode == DEVICE){

        mm_cublas(
            res_grd,
            mat2,
            mat1_grd,
            1,          // alpha = 1
            0,          // beta = 0
            false,      // transpose weights
            true);      //

        mm_cublas(
            mat1,
            res_grd,
            mat2_grd,
            1,          // alpha = 1
            0,          // beta = 0
            true,       // transpose res_grd
            false);     //

    }else{
        ASSERT(false);
    }
}
// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_MM_BP_MM_BP_H_
