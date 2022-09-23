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

#ifndef CUDATEST1_SRC_DATA_SPARSEINPUT_H_
#define CUDATEST1_SRC_DATA_SPARSEINPUT_H_

#include "../misc/config.h"
#include "Align.h"
#include "Matrix.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>     // std::setprecision
#include <iostream>    // std::cout, std::fixed
#include <new>
#include <ostream>

struct SparseInput : public Matrix {

    uint32_t         max_entries_per_column;

    SArray<uint32_t> column_indices {1};

    SparseInput(uint32_t m, uint32_t n, uint32_t max_entries_per_column)
        : Matrix(m, n), max_entries_per_column(max_entries_per_column) {
        column_indices = SArray<uint32_t> {n * (1 + max_entries_per_column)};
        column_indices.mallocGpu();
        column_indices.mallocCpu();
    }

    void set(int input_idx, int index) const {
        auto offset = (max_entries_per_column + 1) * input_idx;
        ASSERT(column_indices.cpu_values->m_data[offset] <= max_entries_per_column);
        column_indices.cpu_values->m_data[offset]++;
        column_indices.cpu_values->m_data[offset + column_indices.cpu_values->m_data[offset]] = index;
    }

    int count(int input_idx){
       auto offset = (max_entries_per_column + 1) * input_idx;
       return column_indices.cpu_values->m_data[offset];
    }

    void clear() {
        for (int i = 0; i < n; i++) {
            column_indices.cpu_values->m_data[i * (max_entries_per_column + 1)] = 0;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const SparseInput& data) {
        os << std::fixed << std::setprecision(0);
        for (int p_i = 0; p_i <= data.max_entries_per_column; p_i++) {
            for (int p_n = 0; p_n < data.n; p_n++) {
                int count = data.column_indices(p_n * (data.max_entries_per_column + 1));
                if (p_i > count) {
                    os << std::setw(11) << "";
                } else {
                    os << std::setw(11)
                       << (int) data.column_indices(p_i + p_n * (data.max_entries_per_column + 1));
                }
            }
            os << "\n";
            if (p_i == 0) {
                for (int p_n = 0; p_n < data.n; p_n++) {
                    os << "-----------";
                }
                os << "\n";
            }
        }
        return os;
    }
};

#endif    // CUDATEST1_SRC_DATA_SPARSEINPUT_H_
