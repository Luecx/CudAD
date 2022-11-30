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
#include <vector>
#include <cstdint>


template<int I, int R>
class ReductionLayer : public LayerInterface {
    public:

    Tape weights {R, 1};
    Tape bias    {I / R, 1};

    LayerInterface* previous;

    ReductionLayer(LayerInterface* previous) : previous(previous){
        ASSERT(previous->getOutputSize() == I);

        // create weights
        weights.values    = DenseMatrix(R, previous->getOutputSize());
        weights.gradients = DenseMatrix(R, previous->getOutputSize());

        // randomise weights
        weights.values.randomiseGaussian(0.0f, (float) sqrt(2.0f / previous->getOutputSize()));

        // upload to gpu
        weights.values.gpuUpload();
        bias   .values.gpuUpload();
    }

    uint32_t getOutputSize() const override { return I / R; }
    uint32_t getInputSize() const override { return previous->getOutputSize(); }
    void     apply() override {
        if(previous->isSparse()){
            ERROR(false);
        }else{
            reduce<DEVICE>(previous->getDenseData().values, weights.values, dense_data.values);
            add_mv<DEVICE>(dense_data.values,
                           bias      .values,
                           dense_data.values);
        }
    }
    void     backprop() override {
        if(previous->isSparse()){
            ERROR(false);
        }else{
            add_mv_bp<DEVICE>(dense_data.gradients,
                              bias      .gradients,
                              dense_data.gradients);
            reduce_bp<DEVICE>(previous->getDenseData().values,
                              previous->getDenseData().gradients,
                              weights.values,
                              dense_data.values,
                              dense_data.gradients);

        }
    }

    std::vector<Tape*> getTunableParameters() override {
        if(is_copy) return {};
        return std::vector<Tape*> {&weights, &bias};
    }
};
#endif    // CUDAD_SRC_LAYER_REDUCTIONLAYER_H_
