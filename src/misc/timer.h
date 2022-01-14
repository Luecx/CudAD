
//
// Created by Luecx on 03.01.2022.
//


#ifndef CUDATEST1__TIMER_H_
#define CUDATEST1__TIMER_H_

#include <stdlib.h>
#include <iostream>
#include <chrono>

class Timer {
    std::chrono::steady_clock::time_point _start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point _end{};
public:
    void tick() {
        _end   = std::chrono::steady_clock::time_point {};
        _start = std::chrono::steady_clock::now();
    }

    void tock() { _end = std::chrono::steady_clock::now();}

    long long duration() const {
//        gsl_Expects(_end != timep_t {} && "toc before reporting");

        return std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
    }
};

#endif //CUDATEST1__TIMER_H_
