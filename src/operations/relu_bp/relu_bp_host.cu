
//
// Created by Luecx on 13.01.2022.
//
#include "relu_bp.h"

#include <iostream>

// clang-format off
void relu_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    unsigned int size){
    // clang-format on

    for (int idx = 0; idx < size; idx++) {
        if (B[idx] > 0) {
            A_grd[idx] = B_grd[idx];
        } else {
            A_grd[idx] = 0;
        }
    }
}
