//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDAD_SRC_DATA_MATRIX_H_
#define CUDAD_SRC_DATA_MATRIX_H_

#include <cstdint>

#define MATRIX_INDEX(ld, m, n) (ld * n + m)

class Matrix{

public:
    uint32_t m;
    uint32_t n;

    Matrix(uint32_t m, uint32_t n) : m(m), n(n) {}
    Matrix(const Matrix& other){
        this->m = other.m;
        this->n = other.n;
    }
    Matrix(Matrix&& other){
        this->m = other.m;
        this->n = other.n;
    }
    Matrix& operator=(const Matrix& other){
        this->m = other.m;
        this->n = other.n;
        return *this;
    }
    Matrix& operator=(Matrix&& other){
        this->m = other.m;
        this->n = other.n;
        return *this;
    }

};

#endif //CUDAD_SRC_DATA_MATRIX_H_
