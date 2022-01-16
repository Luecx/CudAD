#include <iostream>
#include "data/SArray.h"
#include "data/DenseMatrix.h"
#include "operations/operations.h"
#include "misc/timer.h"
#include "layer/DenseLayer.h"
#include "data/Tape.h"
#include "activations/Linear.h"
#include "optimizer/Adam.h"

int main() {

    init();

    constexpr uint32_t I = 1024;
    constexpr uint32_t O = 512;
    constexpr uint32_t B = 16384;

    DenseLayer<I,O, Linear> dense_layer{};

    std::vector<LayerInterface*> layers{};
    layers.push_back(&dense_layer);

    Tape input{I,B};
    input.values.randomise(-1,1);
    DenseMatrix target{I,B};
    target.randomise(-1,1);

    SparseInput sparse_input{I,B,32};
    for(int i = 0; i < B; i++){
        for(int j = 0; j < 16+rand() % 8; j++){
            sparse_input.set(i, rand() % I);
        }
    }
    sparse_input.column_indices.gpu_upload();

    Tape output{O, B};

    input.values.gpu_upload();
    target.gpu_upload();

    Adam adam{};
    adam.alpha = 0.01;
    adam.init(layers);

    std::cout << "======================================================================="<< std::endl;
    Timer t{};
    for(int i = 0; i < 1000; i++){
        dense_layer.apply(sparse_input, output);
//        output.values.gpu_download();
//        float loss = 0;
//        for(int i = 0; i < output.values.size; i++){
//            loss += (output.values(i) -target(i)) * (output.values(i) -target(i));
//        }
//        std::cout << "loss= " << loss << std::endl;

//        add<DEVICE>(output.values,target,output.gradients,1,-1);
        dense_layer.backprop(sparse_input, output);
//        add<DEVICE>(dense_layer.weights.values, dense_layer.weights.gradients, dense_layer.weights.values, 1,-1 / 1e4);
//        adam.apply(1);
    }
    cudaDeviceSynchronize();
    t.tock();
    std::cout << t.duration() << std::endl;

    close();

//    init();
//
//    uint32_t B = 16384;
//    uint32_t I = 768;
//    uint32_t O = 512;
//
//    DenseMatrix wgt    {O,I};
//    DenseMatrix wgt_grd{O,I};
//    DenseMatrix wgt_fm {O,I};
//    DenseMatrix wgt_sm {O,I};
//
//    SparseInput inp{I,B,32};
//
//    DenseMatrix bia    {O,1};
//    DenseMatrix bia_grd{O,1};
//
//    DenseMatrix res    {O,B};
//    DenseMatrix res_grd{O,B};
//
//    DenseMatrix tar{O,B};
//
//    res_grd.randomise(-1,1);
//    for(int b = 0; b < B; b++){
//        for(int k = 0; k < 10 + rand() % 22; k++){
//
//            inp.set(b, rand() % I);
//        }
//    }
//
//    wgt.randomise(-1,1);
//    tar.randomise(-1,1);
//    for(int i = 0; i < res_grd.size; i++){
//        if(i % 2) res_grd(i) = 0;
//    }
//
//    wgt.gpu_upload();
//    bia.gpu_upload();
//    res.gpu_upload();
//    tar.gpu_upload();
//    inp.column_indices.gpu_upload();
//
//    wgt_grd.gpu_upload();
//    bia_grd.gpu_upload();
//    res_grd.gpu_upload();
//
//    Timer t{};
//    for(int i = 0; i < 1000; i++){
////        sparse_affine<DEVICE>(wgt, inp, bia, res);
//        sparse_affine_bp<DEVICE>(wgt_grd, inp, bia_grd, res_grd);
//    }
//    cudaDeviceSynchronize();
//    t.tock();
//    std::cout << t.duration() << std::endl;
//
//    inp.column_indices.gpu_download();
//    wgt.gpu_download();
//    bia.gpu_download();
//    res.gpu_download();
//
////    std::cout << wgt << std::endl;
////    std::cout << bia << std::endl;
////    std::cout << res << std::endl;
//
//    close();
//
////    dense_matrix(3,3) = 1;
////    dense_matrix(3) = 4;
////
////    DenseMatrix test{4,1};
////    for(int i = 0; i < 4; i++){
////        test(i) = i + 1;
////    }
////    std::cout << dense_matrix << " \n " << test << std::endl;
////    test.gpu_upload();
////
////    add<DEVICE>(dense_matrix, test, dense_matrix, 0,0.45);
////    dense_matrix.gpu_download();
////
////    std::cout << dense_matrix << std::endl;
////    for(int i = 0; i < 12; i++){
////        dense_matrix(i,i) = i;
////    }
//
////    DenseMatrix sub_matrix{6,6};
////    sub_matrix(0,0) = 1;
////
////
////    std::cout << dense_matrix.cpu_values << std::endl;
////    std::cout << sub_matrix.cpu_values << std::endl;
////    std::cout << dense_matrix << std::endl;
////    std::cout << sub_matrix << std::endl;

}
