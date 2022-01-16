
//
// Created by Luecx on 13.01.2022.
//
#include <iostream>
#include "add_mv_bp.h"
/**
 * performs C = alpha * A + beta * B
 * @param A
 * @param B
 * @param C
 * @param size
 * @param alpha
 * @param beta
 */
void add_mv_bp_host(
          float* mat_grd,
          float* vec_grd,
    const float* res_grd,
          int m,
          int n,
          int ld_mat,
          int ld_res){

    for(int p_m = 0; p_m < m; p_m++){
        for(int p_n = 0; p_n < n; p_n++){

            float res_grd_v = res_grd[MATRIX_INDEX(ld_res, p_m, p_n)];

            vec_grd[p_m] += res_grd_v;
            mat_grd[MATRIX_INDEX(ld_mat, p_m, p_n)] += res_grd_v;
        }
    }

}
