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

#include "add_mv.h"

#include <iostream>
// clang-format off

void add_mv_host(
    const float* mat,
    const float* vec,
          float* res,
          int m,
          int n,
          int ld_mat,
          int ld_res){
    // clang-format on

    for (int p_m = 0; p_m < m; p_m++) {
        for (int p_n = 0; p_n < n; p_n++) {
            res[MATRIX_INDEX(ld_res, p_m, p_n)] = mat[MATRIX_INDEX(ld_mat, p_m, p_n)] + vec[p_m];
        }
    }
}
