
//
// Created by Luecx on 13.01.2022.
//
#include <iostream>

// clang-format off
void relu_host(
    const float* A,
          float* B,
    unsigned int size){
    // clang-format on

    for (int i = 0; i < size; i++) {
        B[i] = std::max(A[i], 0.0f);
    }
}
