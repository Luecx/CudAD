/****************************************************************************************************
 *                                                                                                  *
 *                                                FFES                                              *
 *                                          by. Finn Eggers                                         *
 *                                                                                                  *
 *                    FFES is free software: you can redistribute it and/or modify                  *
 *                it under the terms of the GNU General Public License as published by              *
 *                 the Free Software Foundation, either version 3 of the License, or                *
 *                                (at your option) any later version.                               *
 *                       FFES is distributed in the hope that it will be useful,                    *
 *                   but WITHOUT ANY WARRANTY; without even the implied warranty of                 *
 *                   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
 *                            GNU General Public License for more details.                          *
 *                 You should have received a copy of the GNU General Public License                *
 *                   along with FFES.  If not, see <http://www.gnu.org/licenses/>.                  *
 *                                                                                                  *
 ****************************************************************************************************/

//
// Created by Luecx on 20.06.2022.
//

#ifndef CUDAD_SRC_LAYER_ACTIVATIONLAYER_H_
#define CUDAD_SRC_LAYER_ACTIVATIONLAYER_H_

#include "Layer.h"
#include "../activations/Linear.h"

template<typename F=Linear>
class ActivationLayer : public LayerInterface {
    public:
    LayerInterface* previous;
    F f;

    ActivationLayer(LayerInterface* previous) : previous(previous){
        ERROR(!previous->isSparse());
    }

    uint32_t getOutputSize() const override { return getInputSize(); }
    uint32_t getInputSize() const override { return previous->getOutputSize(); }
    void     apply() override {
        f.apply(previous->getDenseData().values,
                dense_data.values, DEVICE);
    }
    void     backprop() override {
        f.backprop(previous->getDenseData().values,
                   previous->getDenseData().gradients,
                   dense_data.values,
                   dense_data.gradients, DEVICE);
    }

    std::vector<Tape*> getTunableParameters() override {
        return {};
    }
};


#endif    // CUDAD_SRC_LAYER_ACTIVATIONLAYER_H_
