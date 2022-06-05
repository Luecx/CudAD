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

#ifndef CUDAD_SRC_OPTIMIZER_OPTIMISER_H_
#define CUDAD_SRC_OPTIMIZER_OPTIMISER_H_

#include "../layer/Layer.h"
#include "LRScheduler.h"

/**
 * basic interface for all optimisers.
 * An optimiser has to do the following:
 *
 * Use the gradients to optimise and adjust the values.
 */
struct Optimiser {
    double             lr = 1e-3;
    LRScheduler        schedule {};

    std::vector<Tape*> tunable_values {};

    /**
     * Retrieves the tunable values from all the layers.
     * Will call "createBuffers" which each optimiser needs to use to fill certain fields
     * like first and second moments (adam).
     * @param layers        the layers which the network consists of
     */
    void init(std::vector<LayerInterface*> layers) {
        for (LayerInterface* l : layers) {
            for (auto* k : l->getTunableParameters()) {
                tunable_values.push_back(k);
            }
        }
        createBuffers();
    }

    /**
     * creates buffers for each tape. (e.g. first and second moments)
     */
    virtual void createBuffers() = 0;

    /**
     * The function is supposed to go through the gradients of the weights and biases stored
     * in tunable_values.
     * Furthermore the gradients must be cleared in the process!
     * Beside the gradients, an adjustment of the learning rate can be done based on the amount of
     * batches which have had impact on the gradients.
     * @param td            ThreadData which contains the gradients
     * @param batch_size    batch size which has been used
     */
    virtual void apply(int batch_size) = 0;

    /**
     * If the optimiser requires information about when a new epoch starts, this function can be used.
     * This is mostly unused.
     */
    virtual void newEpoch() = 0;

    /**
     * Used to display information to the log file
     */
    virtual void logOverview() = 0;
};

#endif    // CUDAD_SRC_OPTIMIZER_OPTIMISER_H_
