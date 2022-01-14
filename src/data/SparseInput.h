

//
// Created by Luecx on 08.01.2022.
//

#ifndef CUDATEST1_SRC_DATA_SPARSEINPUT_H_
#define CUDATEST1_SRC_DATA_SPARSEINPUT_H_

#include "../config/config.h"
#include "align.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>     // std::setprecision
#include <iostream>    // std::cout, std::fixed
#include <new>
#include <ostream>

// CSC format
struct SparseInput : public Matrix {

    uint32_t         max_entries_per_column;

    SArray<uint32_t> column_indices {1};

    SparseInput(uint32_t m, uint32_t n, uint32_t max_entries_per_column)
        : Matrix(m, n), max_entries_per_column(max_entries_per_column) {
        column_indices = SArray<uint32_t> {n * (1 + max_entries_per_column)};
        column_indices.malloc_gpu();
        column_indices.malloc_cpu();
    }

    void set(int input_idx, int index) {
        auto offset = (max_entries_per_column + 1) * input_idx;
        column_indices.cpu_values[offset]++;
        column_indices.cpu_values[offset + column_indices.cpu_values[offset]] = index;
    }

    friend std::ostream& operator<<(std::ostream& os, const SparseInput& data) {
        os << std::fixed << std::setprecision(0);
        for (int p_i = 0; p_i <= data.max_entries_per_column; p_i++) {
            for (int p_n = 0; p_n < data.n; p_n++) {
                os << std::setw(11)
                   << (int) data.column_indices(p_i + p_n * (data.max_entries_per_column + 1));
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
