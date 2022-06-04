
//
// Created by Luecx on 13.01.2022.
//
#include <cmath>
#include <iostream>
// clang-format off
void clamp_host(
    float* values,
    float p_min,
    float p_max,
    float size){

    // clang-format on
    for (int idx = 0; idx < size; idx++) {

        if (idx >= size)
            return;

        values[idx] = std::max(p_min, std::min(p_max, values[idx]));
    }
}
