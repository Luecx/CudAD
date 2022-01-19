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
#include "position/position.h"
#include "position/fenparsing.h"
#include "dataset/dataset.h"
#include "dataset/io.h"
#include "dataset/reader.h"
#include "dataset/batchloader.h"
#include "mappings.h"

int main() {

    init();


    constexpr uint32_t I = 768;
    constexpr uint32_t H = 512;
    constexpr uint32_t O = 1;
    constexpr uint32_t B = 16384;

    const std::string     path         = "../resources/networks/koi7.9_half_kingside_512_mirror/";
    const std::string     data_path    = path + "data/";
    const std::string     network_path = path + "networks/";

    // ----------------------------------------------- LOADING DATA ------------------------------------------------------------
    std::vector<std::string> files {};
    for(int i = 1; i < 96; i++){
        files.push_back(R"(H:\Koivisto Resourcen\Training Data 7.9\shuffled_generated_)" + std::to_string(i) + ".txt.bin");
    }
    DataSet validation_set = read<BINARY>(R"(H:\Koivisto Resourcen\Training Data 7.9\shuffled_generated_0.txt.bin)");

    BatchLoader batch_loader{files, B, 1};

    SparseInput sparse_input_1{I,B,32};
    SparseInput sparse_input_2{I,B,32};
    DenseMatrix target{O,B};
    SArray<float> loss{(uint32_t)1};
    loss.malloc_gpu();
    loss.malloc_cpu();


    // create the layers
    DuplicateDenseLayer<I,H, ReLU> l1{};
    DenseLayer<H*2,O, Sigmoid> l2{};

    std::vector<LayerInterface*> layers{};
    layers.push_back(&l1);
    layers.push_back(&l2);
    Sigmoid* act = dynamic_cast<Sigmoid*>(l2.getActivationFunction());
    act->scalar = 2.5 / 400;

    // setup optimiser
    Adam adam{};
    adam.alpha = 0.01 ;
    adam.init(layers);

    // create output and input buffer
    Tape        hiddenOutput{2*H,B};
    Tape        output{O,B};


    Timer t{};
    for(int i = 0; i < 1e6; i++){

        // get the next dataset (batch)
        auto* ds = batch_loader.next();
        // assign to the inputs and compute the target
        dense_relative::assign_inputs_batch(*ds, sparse_input_1, sparse_input_2, target);
        // upload relevant data
        sparse_input_1.column_indices .gpu_upload();
        sparse_input_2.column_indices .gpu_upload();
        target                        .gpu_upload();

        // download the loss to display the loss of the previous iteration
        loss.gpu_download();
        // measure time
        t.tock();
        std::cout << "\rbatch = " << i << " loss = " << loss.cpu_values[0] << " speed = " << 1000.0f * (B * (i+1) / (float)t.duration()) << std::flush;
        // make sure to reset the loss here since the mse increments the loss in order to not have to use memcpy (might change soon)
        loss(0) = 0;
        loss.gpu_upload();

        // feed forward
        l1.apply(sparse_input_1, sparse_input_2, hiddenOutput);
        l2.apply(hiddenOutput, output);

        // compute loss
        mse<DEVICE>(output.values, output.gradients, target, loss);

        // backward
        l2.backprop(hiddenOutput, output);
        l1.backprop(sparse_input_1, sparse_input_2, hiddenOutput);

        // update weights
        adam.apply(1);
    }
    cudaDeviceSynchronize();
    t.tock();
    std::cout << t.duration() << std::endl;
//
    close();

}
