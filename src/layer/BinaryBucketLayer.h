
//
// Created by Luecx on 30.01.2022.
//

#ifndef CUDAD_SRC_LAYER_BINARYBUCKETLAYER_H_
#define CUDAD_SRC_LAYER_BINARYBUCKETLAYER_H_

#include "Layer.h"
#include "../activations/Linear.h"
#include "../operations/operations.h"

template<int I, int B>
class BinaryBucketLayer : public LayerInterface{
    public:

    float max_lower_bucket;
    float min_upper_bucket;
    Linear    f                         {};

    BinaryBucketLayer(float max_lower_bucket, float min_upper_bucket)
        : max_lower_bucket(max_lower_bucket), min_upper_bucket(min_upper_bucket) {}

    void apply(std::vector<Tape*> inputs, Tape &out) override{
       bucket<DEVICE>(inputs[0]->values, out.values, max_lower_bucket, min_upper_bucket);
    }
    void apply(std::vector<SparseInput*> inputs, Tape &out) override{
        ASSERT(false);
    }

    void backprop(std::vector<Tape*> inputs, Tape& out) override {
        bucket_bp<DEVICE>(inputs[0]->values, inputs[0]->gradients, out.gradients, max_lower_bucket, min_upper_bucket);
    }
    void backprop(std::vector<SparseInput*> inputs, Tape& out) override {
        ASSERT(false);
    }

    uint32_t  getOutputSize() override {
        return I*B;
    }
    uint32_t  getInputSize() override {
        return I;
    }
    std::vector<Tape*> getTunableParameters() override {
        std::vector<Tape*> values{};
        return values;
    }
    Activation* getActivationFunction() override { return &f; }
};


#endif    // CUDAD_SRC_LAYER_BINARYBUCKETLAYER_H_
