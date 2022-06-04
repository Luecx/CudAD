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

#include "config.h"

#include <iostream>

cublasHandle_t CUBLAS_HANDLE;

void           init() {
    cublasCreate(&CUBLAS_HANDLE);
    display_header();
}

void display_header() {
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "C++ version:    v" << __cplusplus << std::endl;
    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;

    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        // clang-format off
        std::cout << (i+1)       << ": "
                  << props.name  << ": "
                  << props.major << "."
                  << props.minor << std::endl;
        std::cout << "  Global memory:          " << props.totalGlobalMem    / mb << "mb" << std::endl;
        std::cout << "  Shared memory:          " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::cout << "  Constant memory:        " << props.totalConstMem     / kb << "kb" << std::endl;
        std::cout << "  Block registers:        " << props.regsPerBlock << std::endl << std::endl;

        std::cout << "  Warp size:              " << props.warpSize                 << std::endl;
        std::cout << "  Threads per block:      " << props.maxThreadsPerBlock       << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", "
                                                  << props.maxThreadsDim[1] << ", "
                                                  << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize  [0] << ", "
                                                  << props.maxGridSize  [1] << ", "
                                                  << props.maxGridSize  [2] << " ]" << std::endl;
        // clang-format on
        std::cout << std::endl;
    }
}

void close() { cublasDestroy(CUBLAS_HANDLE); }
