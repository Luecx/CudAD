//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_DENSELAYER_H
#define DIFFERENTIATION_DENSELAYER_H

#include "../misc/config.h"
#include "Layer.h"
#include "../operations/operations.h"
#include "../data/Tape.h"

#include <cmath>

template<int I, int O, typename F>
class DenseLayer : public LayerInterface{
public:
    Tape tunable_values            {O, I + 1};                  // update weights + biases at the same
    Tape weights                   {tunable_values, 0, 0, O,I}; // time
    Tape bias                      {tunable_values, 0, I, O, 1};
    F    f                         {};

    DenseLayer() {
        weights.values.randomiseGaussian(0.0f, (float)sqrt(2.0f / I));
        weights.values.gpu_upload();
        bias   .values.gpu_upload();
    }

    void apply(Tape &in, Tape &out) override{
        mm<DEVICE>(weights.values, in.values, out.values);
        add_mv<DEVICE>(out.values, bias.values, out.values);
        f.apply(out.values, out.values, DEVICE);
    }
    void apply(SparseInput &in, Tape &out) override{
        sparse_affine<DEVICE>(weights.values, in, bias.values, out.values);
        f.apply(out.values, out.values, DEVICE);
    }

    void backprop(Tape &input, Tape& out) override {
        // hack (out.values is not the actual input but the output)
        // this works for the current activation functions since they do not use that value
        f.backprop(out.values, out.gradients, out.values, out.gradients, DEVICE);
        mm_bp<DEVICE>(weights.values, weights.gradients, input.values, input.gradients, out.gradients);
    }
    void backprop(SparseInput &input, Tape& out) override {
        // hack (out.values is not the actual input but the output)
        // this works for the current activation functions since they do not use that value
        f.backprop(out.values, out.gradients, out.values, out.gradients, DEVICE);
        sparse_affine_bp<DEVICE>(weights.gradients, input, bias.gradients, out.gradients);
    }

    int  getOutputSize() override {
        return O;
    }
    int  getInputSize() override {
        return I;
    }
    std::vector<Tape*> getTunableParameters() override {
        std::vector<Tape*> values{};
        values.push_back(&tunable_values);
        return values;
    }
    //    DenseMatrix *getBias() override {
//        return &bias;
//    }
//    DenseMatrix *getWeights() override {
//        return &weights;
//    }
//    DenseMatrix newWeightInstance() override {
//        return weights.newInstance();
//    }
//    DenseMatrix newBiasInstance() override {
//        return bias.newInstance();
//    }
    Activation* getActivationFunction() override { return &f; }
};


#endif //DIFFERENTIATION_DENSELAYER_H
