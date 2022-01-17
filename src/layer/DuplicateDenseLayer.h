//
// Created by Luecx on 25.02.2021.
//

#ifndef DIFFERENTIATION_DUPLICATEDENSELAYER_H
#define DIFFERENTIATION_DUPLICATEDENSELAYER_H

#include "Layer.h"
#include "../operations/operations.h"

template<int I, int O, typename F>
class DuplicateDenseLayer : public LayerInterface{
    public:
    Tape tunable_values            {O, I + 1};                  // update weights + biases at the same
    Tape weights                   {tunable_values, 0, 0, O,I}; // time
    Tape bias                      {tunable_values, 0, I, O, 1};
    F           f {};

    explicit DuplicateDenseLayer(int expected_active_inputs=I) {
        weights.values.randomiseGaussian(0, 2.0f / sqrtf((float)expected_active_inputs));

        weights.values.gpu_upload();
        bias   .values.gpu_upload();
    }

    void apply(Tape &in, Tape &out) override{
//        mat_mat_product<MODE_GPU>(weights, in, out);
//        vm_add<MODE_GPU>(out, bias, out);
//        f.apply(out, out, MODE_GPU);
        ASSERT(false);
    }
    void apply(SparseInput &in, Tape &out) override{
//        bin_sparse_mm_prod_half<MODE_GPU>(weights, in, bias, out);
//        vm_add<MODE_GPU>(out, bias, out);
//        f.apply(out, out, MODE_GPU);
        ASSERT(false);
    }
    void apply(SparseInput &in1,SparseInput &in2, Tape &out){



        uint32_t B = out.values.n;

        // create submatrices for the output
        DenseMatrix mat_res_1{out.values, 0,0,O,B};
        DenseMatrix mat_res_2{out.values, O,0,O,B};

        sparse_affine<DEVICE>(weights.values, in1, bias.values, mat_res_1);
        sparse_affine<DEVICE>(weights.values, in2, bias.values, mat_res_2);
//        //        bin_sparse_mm_prod_half<MODE_GPU>(weights, in, bias, out);
//        //        vm_add<MODE_GPU>(out, bias, out);
        f.apply(out.values, out.values, DEVICE);
    }

    void backprop(SparseInput &in1,SparseInput &in2, Tape &out){
        uint32_t B = out.values.n;

        f.backprop(out.values, out.gradients,out.values, out.gradients, DEVICE);

        // create submatrices for the output
        DenseMatrix mat_res_1{out.gradients, 0,0,O,B};
        DenseMatrix mat_res_2{out.gradients, O,0,O,B};

        sparse_affine_bp<DEVICE>(weights.gradients, in1, bias.gradients, mat_res_1);
        sparse_affine_bp<DEVICE>(weights.gradients, in2, bias.gradients, mat_res_2);
        //        //        bin_sparse_mm_prod_half<MODE_GPU>(weights, in, bias, out);
        //        //        vm_add<MODE_GPU>(out, bias, out);
    }

    void backprop(Tape &input, Tape& out) override {
//        f.backprop(out,out, MODE_GPU);
//        vm_add_backprop<MODE_GPU>(out, bias, out);
//        mat_mat_product_backprop<MODE_GPU>(weights, input, out);
        ASSERT(false);
    }
    void backprop(SparseInput &input, Tape& out) override {
//        f.backprop(out,out, MODE_GPU);
//        vm_add_backprop<MODE_GPU>(out, bias, out);
//        bin_sparse_mm_prod_half_backprop<MODE_GPU>(weights, input, bias, out);
        ASSERT(false);
    }

//    void apply(Data *input, ThreadData *td) override {
//
//    }
//    void backprop(Data *input, ThreadData *td) override {
//
//    }
//
//    void apply(Input    *in1,
//               ThreadData* td){
//        matmul(&weights, in1, im1[td->threadID], 0);
//        matmul(&weights, in1, im2[td->threadID], I);
//        im1[td->threadID]->add(&bias);
//        im2[td->threadID]->add(&bias);
//        f.apply(im1[td->threadID], im1[td->threadID]);
//        f.apply(im2[td->threadID], im2[td->threadID]);
//    }
//
//    void backprop(
//            Input      *in1,
//            ThreadData *td){
//        f.backprop(im1[td->threadID], im1_g[td->threadID],im1_g[td->threadID]);
//        f.backprop(im2[td->threadID], im2_g[td->threadID],im2_g[td->threadID]);
//
//        *td->bias_gradient[layerID] =   *im1_g[td->threadID];
//         td->bias_gradient[layerID]->add(im2_g[td->threadID]);
//
//        matmul_backprop(in1, td->weight_gradient[layerID], im1_g[td->threadID], 0);
//        matmul_backprop(in1, td->weight_gradient[layerID], im2_g[td->threadID], I);
//    }
//

    int  getOutputSize() override {
        return O*2;
    }
    int  getInputSize() override {
        return I*2;
    }
    std::vector<Tape*> getTunableParameters() override {
        std::vector<Tape*> values{};
        values.push_back(&tunable_values);
        return values;
    }
//    DenseMatrix newWeightInstance() override {
//        return weights.newInstance();
//    }
//    DenseMatrix newBiasInstance() override {
//        return bias.newInstance();
//    }
    Activation* getActivationFunction() override { return &f; }
};

#endif //DIFFERENTIATION_DUPLICATEDENSELAYER_H
