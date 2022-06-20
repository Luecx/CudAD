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

    PairwiseMultiplyLayer() {}

//    void apply(std::vector<Tape*> inputs, Tape& out) override {
//        pairwise_multiply<DEVICE>(inputs[0]->values, out.values);
//    }
//    void apply(std::vector<SparseInput*> inputs, Tape& out) override { ASSERT(false); }
//
//    void backprop(std::vector<Tape*> inputs, Tape& out) override {
//        pairwise_multiply_bp<DEVICE>(inputs[0]->values, inputs[0]->gradients, out.gradients);
//    }
//    void     backprop(std::vector<SparseInput*> inputs, Tape& out) override { ASSERT(false); }
//
//    uint32_t getOutputSize() override { return I / 2; }
//    uint32_t getInputSize() override { return I; }


    std::vector<Tape*> getTunableParameters() override { return std::vector<Tape*> {}; }
};

#endif    // CUDAD_SRC_LAYER_PAIRWISEMULTIPLYLAYER_H_
