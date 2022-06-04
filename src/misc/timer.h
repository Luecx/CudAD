/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUDATEST1__TIMER_H_
#define CUDATEST1__TIMER_H_

#include <chrono>
#include <iostream>
#include <stdlib.h>

class Timer {
    std::chrono::steady_clock::time_point _start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point _end {};

    public:
    void tick() {
        _end   = std::chrono::steady_clock::time_point {};
        _start = std::chrono::steady_clock::now();
    }

    void      tock() { _end = std::chrono::steady_clock::now(); }

    long long duration() const {
        //        gsl_Expects(_end != timep_t {} && "toc before reporting");

        return std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
    }
};

#endif    // CUDATEST1__TIMER_H_
