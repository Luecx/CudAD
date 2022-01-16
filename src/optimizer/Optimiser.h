//
// Created by Luecx on 16.01.2022.
//

#ifndef CUDAD_SRC_OPTIMIZER_OPTIMISER_H_
#define CUDAD_SRC_OPTIMIZER_OPTIMISER_H_

#include "../layer/Layer.h"
/**
 * basic interface for all optimisers.
 * An optimiser has to do the following:
 *
 * Use the gradients to optimise and adjust the values.
 */
struct Optimiser {

    std::vector<Tape*> tunable_values {};

    /**
     * Retrieves the tunable values from all the layers.
     * Will call "createBuffers" which each optimiser needs to use to fill certain fields
     * like first and second moments (adam).
     * @param layers        the layers which the network consists of
     */
    void               init(std::vector<LayerInterface*> layers) {
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
    virtual void createBuffers()       = 0;

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
    virtual void newEpoch()            = 0;

    /**
     * Used to display information to the log file
     */
    virtual void logOverview()         = 0;
};

#endif    // CUDAD_SRC_OPTIMIZER_OPTIMISER_H_
