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
#include <vector>
#include <cstdint>

template<int O>
class DenseLayer : public LayerInterface {
    public:
    Tape weights {O, 1};
    Tape bias    {O, 1};
    float lasso_regularization = 0;
    bool is_copy = false;
    LayerInterface* previous;

    DenseLayer(LayerInterface* previous) : previous(previous){
        // create weights
        weights.values    = DenseMatrix(O, previous->getOutputSize());
        weights.gradients = DenseMatrix(O, previous->getOutputSize());

        // randomise weights
        weights.values.randomiseGaussian(0.0f, (float) sqrt(2.0f / previous->getOutputSize()));

        // upload to gpu
        weights.values.gpuUpload();
        bias   .values.gpuUpload();
    }

    DenseLayer(LayerInterface* previous, DenseLayer<O>* copy) : previous(previous){
        // use the same values as the other layer by moving
        weights.values    = std::move(copy->weights.values);
        weights.gradients = std::move(copy->weights.gradients);
        bias.values       = std::move(copy->bias.values);
        bias.gradients    = std::move(copy->bias.gradients);
        is_copy     = true;
    }

    uint32_t getOutputSize() const override { return O; }
    uint32_t getInputSize() const override { return previous->getOutputSize(); }
    void     apply() override {
        if(previous->isSparse()){
            sparse_affine<DEVICE>(weights   .values,
                                  previous->getSparseData(),
                                  bias      .values,
                                  dense_data.values);

        }else{
            mm<DEVICE>(weights                 .values,
                       previous->getDenseData().values,
                       dense_data              .values);


            add_mv<DEVICE>(dense_data.values,
                           bias      .values,
                           dense_data.values);
        }
    }
    void     backprop() override {
        if(previous->isSparse()){
            sparse_affine_bp<DEVICE>(weights  .gradients,
                                     previous->getSparseData(),
                                     bias      .gradients,
                                     dense_data.values,
                                     dense_data.gradients,
                                     lasso_regularization);

        }else{
            add_mv_bp<DEVICE>(dense_data.gradients,
                              bias      .gradients,
                              dense_data.gradients);
            mm_bp<DEVICE>(weights                   .values,
                          weights                   .gradients,
                          previous->getDenseData()  .values,
                          previous->getDenseData()  .gradients,
                          dense_data                .gradients);

        }
    }

    std::vector<Tape*> getTunableParameters() override {
        if(is_copy) return {};
        return std::vector<Tape*> {&weights, &bias};
    }
};

#endif    // DIFFERENTIATION_DENSELAYER_H
