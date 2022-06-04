
//
// Created by Luecx on 28.05.2022.
//

#ifndef CUDAD_SRC_LAYER_PAIRWISEMULTIPLYLAYER_H_
#define CUDAD_SRC_LAYER_PAIRWISEMULTIPLYLAYER_H_

#include "../activations/Linear.h"
#include "../operations/operations.h"
#include "Layer.h"

template<int I>
class PairwiseMultiplyLayer : public LayerInterface {
    public:
    Linear f {};

    PairwiseMultiplyLayer() {}

    void apply(std::vector<Tape*> inputs, Tape& out) override {
        pairwise_multiply<DEVICE>(inputs[0]->values, out.values);
    }
    void apply(std::vector<SparseInput*> inputs, Tape& out) override { ASSERT(false); }

    void backprop(std::vector<Tape*> inputs, Tape& out) override {
        pairwise_multiply_bp<DEVICE>(inputs[0]->values, inputs[0]->gradients, out.gradients);
    }
    void     backprop(std::vector<SparseInput*> inputs, Tape& out) override { ASSERT(false); }

    uint32_t getOutputSize() override { return I / 2; }
    uint32_t getInputSize() override { return I; }
    std::vector<Tape*> getTunableParameters() override {
        std::vector<Tape*> values {};
        return values;
    }
    Activation* getActivationFunction() override { return &f; }
};

#endif    // CUDAD_SRC_LAYER_PAIRWISEMULTIPLYLAYER_H_
