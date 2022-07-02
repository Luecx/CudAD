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

#ifndef CUDAD_SRC_LAYER_PAIRWISEMULTIPLYLAYER_H_
#define CUDAD_SRC_LAYER_PAIRWISEMULTIPLYLAYER_H_

#include "../activations/Linear.h"
#include "../operations/operations.h"
#include "Layer.h"

#include <cmath>
#include <vector>
#include <cstdint>

class PairwiseMultiplyLayer : public LayerInterface {
    public:

    LayerInterface* previous;

    PairwiseMultiplyLayer(LayerInterface* previous) : previous(previous){}

    uint32_t getOutputSize() const override { return getInputSize() / 2; }
    uint32_t getInputSize() const override { return previous->getOutputSize(); }
    void     apply() override {
        pairwise_multiply<DEVICE>(previous->getDenseData().values, dense_data.values);
    }
    void     backprop() override {
        pairwise_multiply_bp<DEVICE>(previous->getDenseData().values,
                                     previous->getDenseData().gradients,
                                     dense_data.gradients);
    }

    std::vector<Tape*> getTunableParameters() override { return std::vector<Tape*> {}; }
};

#endif    // CUDAD_SRC_LAYER_PAIRWISEMULTIPLYLAYER_H_
