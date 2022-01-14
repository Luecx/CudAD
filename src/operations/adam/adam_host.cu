
//
// Created by Luecx on 13.01.2022.
//
#include <iostream>
#include <cmath>
void adam_host(
              float* values,
              float* gradients,
              float* first_moment,
              float* second_moment,
              int   size,
              float alpha,
              float beta1,
              float beta2,
              float eps){

    for(int idx = 0; idx < size; idx++){

        if(idx >= size) return;

        first_moment[idx]  = beta1 * first_moment[idx]  + (1 - beta1) * gradients[idx];
        second_moment[idx] = beta2 * second_moment[idx] + (1 - beta2) * gradients[idx] * gradients[idx];

        float delta = alpha * first_moment[idx] / (sqrtf(second_moment[idx]) + eps);
        values[idx] -= delta;
        gradients[idx] = 0;
    }

}
