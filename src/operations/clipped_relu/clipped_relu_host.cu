
//
// Created by Luecx on 13.01.2022.
//
#include <cmath>
#include <iostream>

// clang-format off
void clipped_relu_host(
    const float* A,
          float* B,
    unsigned int size,
    float max){
    // clang-format on

    for (int i = 0; i < size; i++) {
        B[i] = std::min(max, std::max(A[i], 0.0f));
    }
}
