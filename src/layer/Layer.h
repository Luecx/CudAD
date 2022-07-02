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

#ifndef DIFFERENTIATION_LAYER_H
#define DIFFERENTIATION_LAYER_H

#include "../activations/Activation.h"
#include "../data/DenseMatrix.h"
#include "../data/SparseInput.h"
#include "../data/Tape.h"

#include <cstdint>
#include <vector>

struct LayerInterface {
    public:
    // depending on if this layer is sparse or dense, we have two different output buffers
    SparseInput sparse_data {1, 1, 1};
    Tape        dense_data {1, 1};

    // creates the output tape
    virtual bool isSparse() const { return false; }
    virtual void createOutput(uint32_t batch_size) {

        if (isSparse()) {

            if (sparse_data.n != batch_size || sparse_data.m != getOutputSize()) {
                sparse_data =
                    SparseInput(getOutputSize(), batch_size, sparse_data.max_entries_per_column);
            }
        } else {
            if (dense_data.values.n != batch_size || dense_data.values.m != getOutputSize()) {
                dense_data.values    = DenseMatrix {getOutputSize(), batch_size};
                dense_data.gradients = DenseMatrix {getOutputSize(), batch_size};
            }
        }
    }
    SparseInput& getSparseData() { return sparse_data; }
    Tape&        getDenseData() { return dense_data; }

    virtual uint32_t           getOutputSize() const  = 0;
    virtual uint32_t           getInputSize() const   = 0;

    virtual std::vector<Tape*> getTunableParameters() = 0;

    virtual void               apply()                = 0;
    virtual void               backprop()             = 0;
};

#endif    // DIFFERENTIATION_LAYER_H
