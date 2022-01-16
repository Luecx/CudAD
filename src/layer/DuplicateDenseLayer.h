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
    DenseMatrix      weights             {O,I};
    DenseMatrix      bias                {O};
    F         f                          {};
//    Data*     im1[NN_THREADS]            {};
//    Data*     im2[NN_THREADS]            {};
//    Data*     im1_g[NN_THREADS]          {};
//    Data*     im2_g[NN_THREADS]          {};

    DuplicateDenseLayer(int expected_active_inputs=I) {
        weights.values.randomiseGaussian(0, 2.0 / sqrt(expected_active_inputs));
        bias   .values.randomiseGaussian(0,0);

        weights.values.gpu_upload();
        bias   .values.gpu_upload();
    }

    void apply(DenseMatrix &in, DenseMatrix &out) override{
//        mat_mat_product<MODE_GPU>(weights, in, out);
//        vm_add<MODE_GPU>(out, bias, out);
//        f.apply(out, out, MODE_GPU);
    }
    void apply(SparseInput &in, DenseMatrix &out) override{
//        bin_sparse_mm_prod_half<MODE_GPU>(weights, in, bias, out);
//        vm_add<MODE_GPU>(out, bias, out);
//        f.apply(out, out, MODE_GPU);
    }

    void backprop(DenseMatrix &input, DenseMatrix& out) {
//        f.backprop(out,out, MODE_GPU);
//        vm_add_backprop<MODE_GPU>(out, bias, out);
//        mat_mat_product_backprop<MODE_GPU>(weights, input, out);
    }
    void backprop(SparseInput &input, DenseMatrix& out) {
//        f.backprop(out,out, MODE_GPU);
//        vm_add_backprop<MODE_GPU>(out, bias, out);
//        bin_sparse_mm_prod_half_backprop<MODE_GPU>(weights, input, bias, out);
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
    DenseMatrix *getBias() override {
        return &bias;
    }
    DenseMatrix *getWeights() override {
        return &weights;
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
