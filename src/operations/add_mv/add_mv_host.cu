
//
// Created by Luecx on 13.01.2022.
//
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
