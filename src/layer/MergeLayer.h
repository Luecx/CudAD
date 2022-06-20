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
// Created by Luecx on 19.06.2022.
//

#ifndef CUDAD_SRC_LAYER_MERGELAYER_H_
#define CUDAD_SRC_LAYER_MERGELAYER_H_

#include "../data/Tape.h"
#include "../misc/config.h"
#include "../operations/operations.h"
#include "Layer.h"

#include <cmath>
#include <cstdint>
#include <vector>

struct MergeLayer : LayerInterface {

    public:
    LayerInterface* previous_1;
    LayerInterface* previous_2;

    MergeLayer(LayerInterface* previous_1, LayerInterface* previous_2)
        : previous_1(previous_1), previous_2(previous_2) {
        ERROR(previous_1->isSparse() == false);
        ERROR(previous_2->isSparse() == false);
    }

    uint32_t getOutputSize() const override {
        return previous_1->getOutputSize() + previous_2->getOutputSize();
    }
    uint32_t getInputSize() const override { return getOutputSize(); }
    void     apply() override {
        merge<DEVICE>(previous_1->getDenseData().values,
                      previous_2->getDenseData().values,
                      getDenseData().values);
    }
    void backprop() override {
        merge_bp<DEVICE>(previous_1->getDenseData().gradients,
                         previous_2->getDenseData().gradients,
                         getDenseData().gradients);
    }

    std::vector<Tape*> getTunableParameters() override { return {}; }
};

#endif    // CUDAD_SRC_LAYER_MERGELAYER_H_
