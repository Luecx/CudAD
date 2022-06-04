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

#ifndef CUDAD_SRC_DATA_DENSEMATRIX_H_
#define CUDAD_SRC_DATA_DENSEMATRIX_H_

#include "Matrix.h"
#include "SArray.h"

struct DenseMatrix : public Matrix, public SArray<float> {

    uint32_t leading_dimension;

    public:
    DenseMatrix(uint32_t m, uint32_t n) : Matrix(m, n), SArray(m * n), leading_dimension {m} {
        SArray::malloc_gpu();
        SArray::malloc_cpu();
    }

    DenseMatrix(const DenseMatrix& other)
        : Matrix(other.m, other.n), SArray<float>(other), leading_dimension {
                                                              other.leading_dimension} {}

    DenseMatrix(DenseMatrix&& other)
        : Matrix(other.m, other.n), SArray<float>(other), leading_dimension {
                                                              other.leading_dimension} {}

    DenseMatrix(const DenseMatrix& other, uint32_t m_start, uint32_t n_start, uint32_t m, uint32_t n)
        : Matrix(m, n), SArray<float>(other,
                                      MATRIX_INDEX(other.leading_dimension, m_start, n_start),
                                      other.leading_dimension * n),
          leading_dimension {other.leading_dimension} {
              ASSERT(m + m_start <= other.m) ASSERT(n + n_start <= other.n) ASSERT(
                  other.size >= m * n + MATRIX_INDEX(other.leading_dimension, m_start, n_start))}

              DenseMatrix
              & operator=(const DenseMatrix& other) {
        SArray<float>::operator=(other);
        Matrix::       operator=(other);
    }
    DenseMatrix& operator=(DenseMatrix&& other) {
        SArray<float>::operator=(other);
        Matrix::       operator=(other);
    }

    float& get(int p_m, int p_n) const {
        return SArray<float>::get(MATRIX_INDEX(leading_dimension, p_m, p_n));
        ;
    }
    float  operator()(int p_m, int p_n) const { return get(p_m, p_n); }
    float& operator()(int p_m, int p_n) { return get(p_m, p_n); }
    using SArray<float>::get;
    using SArray<float>::operator();

    friend std::ostream& operator<<(std::ostream& os, const DenseMatrix& data) {

        os << "size:       " << data.size << "\n"
           << "gpu_values: " << data.gpu_values << "    cleanUp = [" << data.clean_gpu << "]\n"
           << "cpu_values: " << data.cpu_values << "    cleanUp = [" << data.clean_cpu << "]\n";

        if (data.n != 1) {
            os << std::fixed << std::setprecision(5);
            for (int p_i = 0; p_i < data.m; p_i++) {
                for (int p_n = 0; p_n < data.n; p_n++) {
                    os << std::setw(11) << (double) data(p_i, p_n);
                }
                os << "\n";
            }
        } else {
            os << "(transposed) ";
            for (int n = 0; n < data.m; n++) {
                os << std::setw(11) << (double) data(n, 0);
            }
            os << "\n";
        }
        return os;
    }
};

#endif    // CUDAD_SRC_DATA_DENSEMATRIX_H_
