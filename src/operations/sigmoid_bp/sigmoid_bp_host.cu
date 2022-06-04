
//
// Created by Luecx on 13.01.2022.
//
#include "sigmoid_bp.h"

#include <iostream>

// clang-format off
void sigmoid_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    unsigned int size,
    float scalar){
    // clang-format on

    for (int idx = 0; idx < size; idx++) {
        A_grd[idx] = B_grd[idx] * B[idx] * (1 - B[idx]) * scalar;
    }
}
