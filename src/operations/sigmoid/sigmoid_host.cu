
//
// Created by Luecx on 13.01.2022.
//
#include <cmath>
#include <iostream>

// clang-format off
void sigmoid_host(
    const float* A,
          float* B,
    unsigned int size,
    float scalar){
    // clang-format on

    for (int i = 0; i < size; i++) {
        B[i] = 1.0f / (1.0f + expf(-scalar * A[i]));
    }
}
