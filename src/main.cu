#include "activations/ClippedReLU.h"
#include "activations/Linear.h"
#include "activations/ReLU.h"
#include "activations/Sigmoid.h"
#include "data/DenseMatrix.h"
#include "data/SArray.h"
#include "data/Tape.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "dataset/io.h"
#include "dataset/reader.h"
#include "dataset/writer.h"
#include "layer/DenseLayer.h"
#include "layer/DuplicateDenseLayer.h"
#include "loss/MLE.h"
#include "loss/MPE.h"
#include "loss/MSE.h"
#include "mappings.h"
#include "misc/timer.h"
#include "network/Network.h"
#include "operations/operations.h"
#include "optimizer/Adam.h"
#include "position/fenparsing.h"
#include "position/position.h"
#include "quantitize.h"

#include <filesystem>
#include <iostream>

const std::string data_path = "E:/berserk/training-data/berserk9dev2/finny-data/";
std::string output = "./resources/runs/testing/";

int main() {

    init();

    // definitions
    constexpr uint32_t       I = 12 * 2 * 64;
    constexpr uint32_t       H = 512;
    constexpr uint32_t       O = 1;
    constexpr uint32_t       B = 16384;
    constexpr uint32_t     BPE = 6100;
    constexpr  int32_t       E = 420;

    // Load files
    std::vector<std::string> files {};
    for (int i = 0; i < 7; i++)
        files.push_back(data_path + "berserk9dev2.d9." + std::to_string(i) + ".bin");
    
    BatchLoader  batch_loader {files, B, 1};

    // Input data (perspective)
    SparseInput  i0 {I, B, 32};    // 32 max inputs
    SparseInput  i1 {I, B, 32};

    DenseMatrix  target {O, B};
    SArray<bool> target_mask {O * B};
    target_mask.malloc_cpu();
    target_mask.malloc_gpu();

    // 1536 -> (2x512) -> 1
    DuplicateDenseLayer<I, H, ReLU> l1 {};
    DenseLayer<H * 2, O, Sigmoid>   l2 {};

    // Berserk sigmoid
    dynamic_cast<Sigmoid*>(l2.getActivationFunction())->scalar = 3.68415f / 512;

    // stack layers to build network
    std::vector<LayerInterface*> layers {};
    layers.push_back(&l1);
    layers.push_back(&l2);

    Network network {layers};

    // loss function
    MPE     loss_function {2.5};
    network.setLossFunction(&loss_function);

    // optimizer
    Adam adam {};
    adam.init(layers);
    adam.alpha = 0.01;
    adam.beta1 = 0.95f;
    adam.beta2 = 0.999f;

    logging::open(output + "loss.csv");

    Timer t {};
    for (int epoch = 1; epoch <= E; epoch++) {
        float epoch_loss = 0;
        t.tick();
        for (int batch = 0; batch < BPE; batch++) {

            // get the next dataset (batch)
            auto* ds = batch_loader.next();
            // assign to the inputs and compute the target
            dense_berky::assign_inputs_batch(*ds, i0, i1, target, target_mask);
            // upload relevant data
            i0.column_indices.gpu_upload();
            i1.column_indices.gpu_upload();
            target.gpu_upload();
            target_mask.gpu_upload();

            // download the loss to display the loss of the iteration
            loss_function.loss.gpu_download();

            // measure time and print output
            t.tock();
            std::printf("\rep/ba = [%3d/%4d], ", epoch, batch + 1);
            std::printf("batch_loss = [%1.8f], ", loss_function.loss(0));
            std::printf("epoch_loss = [%1.8f], ", epoch_loss / (batch + 1));
            std::printf("speed = [%9.0f pos/s]", std::round(1000.0f * (B * (batch + 1) / (float) t.duration())));
            std::cout << std::flush;

            epoch_loss += loss_function.loss(0);
            // make sure to reset the loss here since the mse increments the loss in order to not have
            // to use memcpy (might change soon)
            loss_function.loss(0) = 0;
            loss_function.loss.gpu_upload();

            // feed forward
            network.batch(std::vector<SparseInput*> {&i0, &i1},
                        target,
                        target_mask);

            // update weights
            adam.apply(1);
        }
        std::cout << std::endl;
        logging::write("\"" + std::to_string(epoch) + "\",\"" + std::to_string(epoch_loss / BPE) + "\"");

        network.saveWeights(output + "weights-epoch" + std::to_string(epoch) + ".nn");
        quantitize(output + "nn-epoch" + std::to_string(epoch) + ".nnue", network);

        if (epoch % 105 == 0)
            adam.alpha *= 0.1f;
    }

    close();
}
