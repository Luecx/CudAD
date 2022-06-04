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

#ifndef CUDAD_SRC_ASSERT_GPUASSERT_H_
#define CUDAD_SRC_ASSERT_GPUASSERT_H_

#define CUDA_ASSERT(ans)                                                                             \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// inline void gpuAssert(cublasStatus_t code, const char *file, int line, bool abort = true) {
//     if (code != CUBLAS_STATUS_SUCCESS) {
//         switch (code) {
//             case CUBLAS_STATUS_NOT_INITIALIZED:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_NOT_INITIALIZED file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_ALLOC_FAILED:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_ALLOC_FAILED file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_INVALID_VALUE:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_INVALID_VALUE file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_ARCH_MISMATCH:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_ARCH_MISMATCH file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_MAPPING_ERROR:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_MAPPING_ERROR file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_EXECUTION_FAILED:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_EXECUTION_FAILED file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_INTERNAL_ERROR:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_INTERNAL_ERROR file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_NOT_SUPPORTED:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_NOT_SUPPORTED file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//
//             case CUBLAS_STATUS_LICENSE_ERROR:
//                 fprintf(stderr,
//                         "cuBLAS Error: CUBLAS_STATUS_LICENSE_ERROR file: %s line: %d ",
//                         file,
//                         line);
//                 break;
//         }
//         if (abort) exit(code);
//     }
// }

#endif    // CUDAD_SRC_ASSERT_GPUASSERT_H_
