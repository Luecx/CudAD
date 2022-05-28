//
// Created by Luecx on 25.02.2021.
//

#ifndef DIFFERENTIATION_DUPLICATEDENSELAYER_H
#define DIFFERENTIATION_DUPLICATEDENSELAYER_H

#include "../operations/operations.h"
#include "Layer.h"

template<int I, int O, typename F>
class DuplicateDenseLayer : public LayerInterface {
    public:
    Tape tunable_values {O, I + 1};               // update weights + biases at the same
    Tape weights        {tunable_values, 0, 0, O, I};    // time
    Tape bias           {tunable_values, 0, I, O, 1};
    F    f              {};
    // regularization
    float lasso_regularization = 0;

    explicit DuplicateDenseLayer(int expected_active_inputs = I) {
        weights.values.randomiseGaussian(0, 2.0f / sqrtf((float) expected_active_inputs));

        weights.values.gpu_upload();
        bias.values.gpu_upload();
    }

    void apply(std::vector<Tape*> inputs, Tape& out) override {
        //        mat_mat_product<MODE_GPU>(weights, in, out);
        //        vm_add<MODE_GPU>(out, bias, out);
        //        f.apply(out, out, MODE_GPU);
        ASSERT(false);
    }
    void apply(std::vector<SparseInput*> inputs, Tape& out) override {
        uint32_t    B = out.values.n;
        // create submatrices for the output
        DenseMatrix mat_res_1 {out.values, 0, 0, O, B};
        DenseMatrix mat_res_2 {out.values, O, 0, O, B};
        sparse_affine<DEVICE>(weights.values, *inputs[0], bias.values, mat_res_1);
        sparse_affine<DEVICE>(weights.values, *inputs[1], bias.values, mat_res_2);
        f.apply(out.values, out.values, DEVICE);
    }

    void backprop(std::vector<Tape*> inputs, Tape& out) override { ASSERT(false); }

    void backprop(std::vector<SparseInput*> inputs, Tape& out) override {
        uint32_t B = out.values.n;

        f.backprop(out.values, out.gradients, out.values, out.gradients, DEVICE);

        // create submatrices for the output
        DenseMatrix mat_grd_1 {out.gradients, 0, 0, O, B};
        DenseMatrix mat_grd_2 {out.gradients, O, 0, O, B};
        DenseMatrix mat_res_1 {out.values, 0, 0, O, B};
        DenseMatrix mat_res_2 {out.values, O, 0, O, B};

        sparse_affine_bp<DEVICE>(weights.gradients, *inputs[0], bias.gradients, mat_res_1, mat_grd_1, lasso_regularization);
        sparse_affine_bp<DEVICE>(weights.gradients, *inputs[1], bias.gradients, mat_res_2, mat_grd_2, lasso_regularization);
    }

    uint32_t           getOutputSize() override { return O * 2; }
    uint32_t           getInputSize() override { return I * 2; }
    std::vector<Tape*> getTunableParameters() override {
        std::vector<Tape*> values {};
        values.push_back(&tunable_values);
        return values;
    }
    Activation* getActivationFunction() override { return &f; }
};

#endif    // DIFFERENTIATION_DUPLICATEDENSELAYER_H
