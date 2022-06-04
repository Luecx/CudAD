

//
// Created by Luecx on 15.01.2022.
//

#ifndef CUDAD_SRC_DATA_TAPE_H_
#define CUDAD_SRC_DATA_TAPE_H_

#include "Matrix.h"

#include <utility>
struct Tape {

    DenseMatrix values;
    DenseMatrix gradients;

    float       min_allowed_value = std::numeric_limits<float>::min();
    float       max_allowed_value = std::numeric_limits<float>::max();

    Tape(uint32_t m, uint32_t n) : values(DenseMatrix {m, n}), gradients(DenseMatrix {m, n}) {};

    Tape(DenseMatrix values, DenseMatrix gradients)
        : values(std::move(values)), gradients(std::move(gradients)) {}

    Tape(Tape& other, uint32_t m_start, uint32_t n_start, uint32_t m, uint32_t n)
        : values(DenseMatrix(other.values, m_start, n_start, m, n)),
          gradients(DenseMatrix(other.gradients, m_start, n_start, m, n)) {}

    Tape(Tape&& other) : values(std::move(other.values)), gradients(std::move(other.gradients)) {}

    Tape(const Tape& other) : values(other.values), gradients(other.gradients) {}
};
#endif    // CUDAD_SRC_DATA_TAPE_H_
