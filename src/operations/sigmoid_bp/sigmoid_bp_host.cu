
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
void sigmoid_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    unsigned int size,
    float scalar){

    for(int idx = 0; idx < size; idx++){
        A_grd[idx] = B_grd[idx] * B[idx] * (1 - B[idx]) * scalar;
    }
}
