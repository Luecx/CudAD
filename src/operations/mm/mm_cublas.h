
//
// Created by Luecx on 07.01.2022.
//

#ifndef CUDATEST1_SRC_OPERATIONS_MAT_MAT_PRODUCT_MAT_MAT_PRODUCT_CUBLAS_H_
#define CUDATEST1_SRC_OPERATIONS_MAT_MAT_PRODUCT_MAT_MAT_PRODUCT_CUBLAS_H_

#include "../../data/DenseMatrix.h"
#include "../../misc/config.h"
// clang-format off
inline void mm_cublas(
    const DenseMatrix &A,
    const DenseMatrix &B,
          DenseMatrix &C,
    const float alpha = 1,
    const float beta = 0,
    bool transpose_A=false,
    bool transpose_B=false) {
    // clang-format on

    const int m       = C.m;
    const int n       = C.n;
    const int k       = transpose_A ? A.m : A.n;

    int       lda     = A.leading_dimension;
    int       ldb     = B.leading_dimension;
    int       ldc     = C.leading_dimension;

    auto      trans_a = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto      trans_b = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    // clang-format off
    cublasSgemm(CUBLAS_HANDLE, trans_a, trans_b,
                m, n, k, &alpha,
                A.gpu_values, lda,
                B.gpu_values, ldb, &beta,
                C.gpu_values, ldc);
    // clang-format on

    CUDA_ASSERT(cudaPeekAtLastError());

    //    cudaDeviceSynchronize();
}

#endif    // CUDATEST1_SRC_OPERATIONS_MAT_MAT_PRODUCT_MAT_MAT_PRODUCT_CUBLAS_H_
