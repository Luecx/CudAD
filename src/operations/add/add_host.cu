
//
// Created by Luecx on 13.01.2022.
//
#include <iostream>
// clang-format off
/**
 * performs C = alpha * A + beta * B
 * @param A
 * @param B
 * @param C
 * @param size
 * @param alpha
 * @param beta
 */
void add_host(
    const float* A,
    const float* B,
          float* C,
    const unsigned int A_size,
    const unsigned int B_size,
    const unsigned int C_size,
    const unsigned int size,
    const float alpha,
    const float beta){
    // clang-format on

    for (int i = 0; i < size; i++) {
        C[i] = A[i % A_size] * alpha + B[i % B_size] * beta;
    }
}
