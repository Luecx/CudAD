
//
// Created by Luecx on 13.01.2022.
//
#include <iostream>
/**
 * performs C = alpha * A + beta * B
 * @param A
 * @param B
 * @param C
 * @param size
 * @param alpha
 * @param beta
 */
void relu_host(
    const float* A,
          float* B,
    unsigned int size){

    for(int i = 0; i < size; i++){
        B[i] = std::max(A[i], 0.0f);
    }
}
