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
#include "network/Network.h"
#include "quantitize.h"

int main() {

    init();

    constexpr uint32_t       I            = 768 * 2;
    constexpr uint32_t       H            = 512;
    constexpr uint32_t       O            = 1;
    constexpr uint32_t       B            = 16384;

    constexpr int MAXIMUM_EPOCHS          = 1024;
    constexpr int BATCHES_PER_EPOCH       = 16384 * 4;

    const std::string        path         = "../resources/runs/koi7.9_half_kingside_512/";
    const std::string        data_path    = path + "data/";
    const std::string        network_path = path + "networks/";

    // ---------------------------------- LOADING DATA -----------------------------------------------
    std::vector<std::string> files {};
    for (int i = 0; i < 96; i++) {
        files.push_back(R"(H:\Koivisto Resourcen\Training Data 7.9\shuffled_generated_)"
                        + std::to_string(i) + ".txt.bin");
    }
//    DataSet validation_set =
//        read<BINARY>(R"(H:\Koivisto Resourcen\Training Data 7.9\shuffled_generated_0.txt.bin)");

    BatchLoader   batch_loader {files, B, 1};

    SparseInput   sparse_input_1 {I, B, 32};
    SparseInput   sparse_input_2 {I, B, 32};
    DenseMatrix   target {O, B};
    SArray<float> loss {(uint32_t) 1};
    loss.malloc_gpu();
    loss.malloc_cpu();

    // create the layers
    DuplicateDenseLayer<I, H, ReLU> l1 {};
    DenseLayer<H * 2, O, Linear>   l2 {};
//    dynamic_cast<Sigmoid*>(l2.getActivationFunction())->scalar  = 2.5 / 400;

    std::vector<LayerInterface*>    layers {};
    layers.push_back(&l1);
    layers.push_back(&l2);
    // setup optimiser
    Adam adam {};
    adam.alpha = 0.001;
    adam.init(layers);

    Network network{layers};

//    logging::open(path + "log.txt");
//    network.loadWeights(network_path + "113.nn");
//    quantitize(network_path + "113.net", network, 64,256);
//    test_fen(network, "8/8/8/8/8/8/8/8 w - - 0 1", I);
//    exit(-1);
//    test_fen(network, "1rn1r1k1/2q2ppp/4p3/p1B5/Pp2p1P1/4Q1N1/P1PR1P1P/2K1R3 w - - 0 21", I);
//    test_fen(network, "3k4/8/8/5p2/4P3/7Q/8/3K4 w - - 0 1", I);

    Timer t{};
    for(int epoch = 1; epoch <= MAXIMUM_EPOCHS; epoch ++){
        float epoch_loss = 0;
        t.tick();
        for(int batch = 0; batch < BATCHES_PER_EPOCH; batch++){

            // get the next dataset (batch)
            auto* ds = batch_loader.next();
            // assign to the inputs and compute the target
            dense_relative::assign_inputs_batch(*ds, sparse_input_1, sparse_input_2, target);
            // upload relevant data
            sparse_input_1.column_indices .gpu_upload();
            sparse_input_2.column_indices .gpu_upload();
            target                        .gpu_upload();

            // download the loss to display the loss of the iteration
            loss.gpu_download();
            // measure time
            t.tock();
            std::stringstream ss{};
            ss
                      << "    ep/ba = [" << std::setw(4) << epoch << "/" << std::setw(5) << batch << "]"
                      << "    ba. loss  = "  << std::setw(14) << loss.cpu_values[0]
                      << "    ep. loss  = "  << std::setw(14) << epoch_loss / (batch + 1)
                      << "    speed = "  << std::setw(10) << std::round(1000.0f * (B * (batch+1) / (float)t.duration()));
            std::cout << "\r" << ss.str() << std::flush;

            epoch_loss += loss(0);
            // make sure to reset the loss here since the mse increments the loss in order to not have to use memcpy (might change soon)
            loss(0) = 0;
            loss.gpu_upload();

            // feed forward
            network.batch({&sparse_input_1, &sparse_input_2}, target, loss);

            // update weights
            adam.apply(1);
        }
        std::cout << std::endl;
        logging::write("epoch          : " + std::to_string(epoch));
        logging::write("train loss     : " + std::to_string(epoch_loss / BATCHES_PER_EPOCH));
//        network.saveWeights(network_path + std::to_string(epoch) + ".nn");
    }

    close();

}
