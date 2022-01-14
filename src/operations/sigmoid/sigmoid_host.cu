
//
// Created by Luecx on 13.01.2022.
//
#include <iostream>
#include <cmath>
/**
 * performs C = alpha * A + beta * B
 * @param A
 * @param B
 * @param C
 * @param size
 * @param alpha
 * @param beta
 */
void sigmoid_host(
    const float* A,
          float* B,
    unsigned int size,
    float scalar){

    for(int i = 0; i < size; i++){
        B[i] = 1.0f / (1.0f + expf(scalar * A[i]));
    }
}
