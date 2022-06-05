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

#ifndef DIFFERENTIATION_DENSELAYER_H
#define DIFFERENTIATION_DENSELAYER_H

#include "../data/Tape.h"
#include "../misc/config.h"
#include "../operations/operations.h"
#include "Layer.h"

#include <cmath>

template<int I, int O, typename F>
class DenseLayer : public LayerInterface {
    public:
    Tape weights {O, I};
    Tape bias {O, 1};
    F    f {};

    DenseLayer() {
        weights.values.randomiseGaussian(0.0f, (float) sqrt(2.0f / I));
        weights.values.gpu_upload();
        bias.values.gpu_upload();
    }

    void apply(std::vector<Tape*> inputs, Tape& out) override {
        mm<DEVICE>(weights.values, inputs[0]->values, out.values);
        add_mv<DEVICE>(out.values, bias.values, out.values);
        f.apply(out.values, out.values, DEVICE);
    }
    void apply(std::vector<SparseInput*> inputs, Tape& out) override {
        sparse_affine<DEVICE>(weights.values, *inputs[0], bias.values, out.values);
        f.apply(out.values, out.values, DEVICE);
    }

    void backprop(std::vector<Tape*> inputs, Tape& out) override {
        // hack (out.values is not the actual input but the output)
        // this works for the current activation functions since they do not use that value
        f.backprop(out.values, out.gradients, out.values, out.gradients, DEVICE);
        add_mv_bp<DEVICE>(out.gradients, bias.gradients, out.gradients);
        mm_bp<DEVICE>(weights.values,
                      weights.gradients,
                      inputs[0]->values,
                      inputs[0]->gradients,
                      out.gradients);
    }
    void backprop(std::vector<SparseInput*> inputs, Tape& out) override {
        // hack (out.values is not the actual input but the output)
        // this works for the current activation functions since they do not use that value
        f.backprop(out.values, out.gradients, out.values, out.gradients, DEVICE);
        sparse_affine_bp<DEVICE>(weights.gradients,
                                 *inputs[0],
                                 bias.gradients,
                                 out.values,
                                 out.gradients);
    }

    uint32_t           getOutputSize() override { return O; }
    uint32_t           getInputSize() override { return I; }
    std::vector<Tape*> getTunableParameters() override {
        return std::vector<Tape*> {&weights, &bias};
    }
    Activation* getActivationFunction() override { return &f; }
};

#endif    // DIFFERENTIATION_DENSELAYER_H
