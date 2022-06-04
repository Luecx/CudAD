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

#ifndef CUDAD_SRC_DATA_MATRIX_H_
#define CUDAD_SRC_DATA_MATRIX_H_

#include <cstdint>

#define MATRIX_INDEX(ld, m, n) (ld * n + m)

class Matrix {

    public:
    uint32_t m;
    uint32_t n;

    Matrix(uint32_t m, uint32_t n) : m(m), n(n) {}
    Matrix(const Matrix& other) {
        this->m = other.m;
        this->n = other.n;
    }
    Matrix(Matrix&& other) {
        this->m = other.m;
        this->n = other.n;
    }
    Matrix& operator=(const Matrix& other) {
        this->m = other.m;
        this->n = other.n;
        return *this;
    }
    Matrix& operator=(Matrix&& other) {
        this->m = other.m;
        this->n = other.n;
        return *this;
    }
};

#endif    // CUDAD_SRC_DATA_MATRIX_H_
