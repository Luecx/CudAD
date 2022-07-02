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

#ifndef CUDAD_SRC_LAYER_INPUT_H_
#define CUDAD_SRC_LAYER_INPUT_H_

#include "Layer.h"

#include <cmath>
#include <cstdint>
#include <ostream>
#include <vector>

struct Input : LayerInterface {

    const bool is_sparse;
    const int  size;

    Input(bool is_sparse, int size, int max_entries = 1) : is_sparse(is_sparse), size(size) {
        sparse_data.max_entries_per_column = max_entries;
    }

    bool                 isSparse() const override { return is_sparse; }

    uint32_t             getOutputSize() const override { return size; }

    friend std::ostream& operator<<(std::ostream& os, const Input& input) {

        if (input.isSparse()) {
            os << input.sparse_data;
        } else {
            os << input.dense_data.values;
        }

        return os;
    }

    void               apply() override {}
    void               backprop() override {}

    uint32_t           getInputSize() const override { return getOutputSize(); }
    std::vector<Tape*> getTunableParameters() override { return {}; }

};

#endif    // CUDAD_SRC_LAYER_INPUT_H_
