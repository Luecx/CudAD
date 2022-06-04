

//
// Created by Luecx on 07.01.2022.
//

#ifndef CUDATEST1_SRC_CONFIG_CONFIG_H_
#define CUDATEST1_SRC_CONFIG_CONFIG_H_

#include <cublas_v2.h>
#include <cusparse_v2.h>

extern cublasHandle_t CUBLAS_HANDLE;

void                  init();
void                  close();
void                  display_header();

#endif    // CUDATEST1_SRC_CONFIG_CONFIG_H_
