#include <iostream>
#include "data/SArray.h"
#include "data/DenseMatrix.h"
#include "operations/operations.h"
#include "misc/timer.h"
#include "layer/DenseLayer.h"
#include "data/Tape.h"
#include "activations/Linear.h"
#include "optimizer/Adam.h"
#include "layer/DuplicateDenseLayer.h"
#include "activations/Sigmoid.h"
#include "activations/ReLU.h"

int main() {

    init();

    constexpr uint32_t I = 768;
    constexpr uint32_t H = 512;
    constexpr uint32_t O = 1;
    constexpr uint32_t B = 16384;

    // create the layers
    DuplicateDenseLayer<I,H, ReLU> l1{};
    DenseLayer<H*2,O, Sigmoid> l2{};

    std::vector<LayerInterface*> layers{};
    layers.push_back(&l1);
    layers.push_back(&l2);

    // setup optimiser
    Adam adam{};
    adam.alpha = 0.01;
    adam.init(layers);

    // create output and input buffer
    SparseInput sparse_input_1{I,B,32};
    SparseInput sparse_input_2{I,B,32};
    Tape        input{I,B};
    Tape        hiddenOutput{2*H,B};
    Tape        output{O,B};
    DenseMatrix target{O,B};

    input.values.randomise(-1,1);
    target.randomise(0,1);

    for(int i = 0; i < B; i++){
        for(int j = 0; j < 16+rand() % 8; j++){
            sparse_input_1.set(i, rand() % I);
            sparse_input_2.set(i, rand() % I);
        }
    }

    // upload relevant data to the gpu
    input       .values           .gpu_upload();
    target                        .gpu_upload();
    sparse_input_1.column_indices .gpu_upload();
    sparse_input_2.column_indices .gpu_upload();



    Timer t{};
    for(int i = 0; i < 1000; i++){
        l1.apply(sparse_input_1, sparse_input_2, hiddenOutput);
        l2.apply(hiddenOutput, output);

//        hiddenOutput.values.gpu_download();


//        output.values.gpu_download();
//        float loss = 0;
//        for(int i = 0; i < output.values.size; i++){
//            loss += (output.values(i) -target(i)) * (output.values(i) -target(i));
//        }
//        std::cout << "loss= " << loss << std::endl;

        add<DEVICE>(output.values,target,output.gradients,1,-1);

        l2.backprop(hiddenOutput, output);
        l1.backprop(sparse_input_1, sparse_input_2, hiddenOutput);
        adam.apply(1);
    }
    cudaDeviceSynchronize();
    t.tock();
    std::cout << t.duration() << std::endl;

    close();

}
