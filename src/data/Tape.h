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

#ifndef CUDAD_SRC_DATA_TAPE_H_
#define CUDAD_SRC_DATA_TAPE_H_

#include "DenseMatrix.h"

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
