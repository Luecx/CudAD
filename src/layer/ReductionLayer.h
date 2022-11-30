/**
AD is a general CUDA neural network framework.
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

//
// Created by Luecx on 29.11.2022.
//

#ifndef CUDAD_SRC_LAYER_REDUCTIONLAYER_H_
#define CUDAD_SRC_LAYER_REDUCTIONLAYER_H_

#include "../data/Tape.h"
#include "../misc/config.h"
#include "../operations/operations.h"
#include "Layer.h"

#include <cmath>
#include <cstdint>
#include <vector>

template<int I, int R, typename F>
class ReductionLayer : public LayerInterface {
    public:
    Tape            weights {I, 1};
    Tape            bias {I / R, 1};
    F    f {};

    ReductionLayer() {
        weights.values.randomise(1.0f, 1.0f);
        weights.values.gpu_upload();
        bias.values.gpu_upload();
    }

    uint32_t           getOutputSize() override { return I / R; }
    uint32_t           getInputSize() override { return I; }
    std::vector<Tape*> getTunableParameters() override {
        return std::vector<Tape*> {};
    }
    Activation* getActivationFunction() override { return &f; }
    void               apply(std::vector<Tape*> inputs, Tape& out) override {
        reduce<DEVICE>(inputs[0]->values, weights.values, out.values);
        add_mv<DEVICE>(out.values, bias.values, out.values);
        f.apply(out.values, out.values, DEVICE);
    }
    void               backprop(std::vector<Tape*> inputs, Tape& out) override {
        f.backprop(out.values, out.gradients, out.values, out.gradients, DEVICE);
        add_mv_bp<DEVICE>(out.gradients, bias.gradients, out.gradients);
        reduce_bp<DEVICE>(inputs[0]->values, inputs[0]->gradients,  weights.values, weights.gradients, out.gradients);
    }
    void               apply(std::vector<SparseInput*> inputs, Tape& out) override {
        // dont support sparse inputs
    }
    void               backprop(std::vector<SparseInput*> inputs, Tape& out) override {
        // dont support sparse inputs
    }

};
#endif    // CUDAD_SRC_LAYER_REDUCTIONLAYER_H_
