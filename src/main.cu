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

int main() {

    init();

    // definitions
    constexpr uint32_t       I = 768 * 2;
    constexpr uint32_t       H = 512;
    constexpr uint32_t       O = 1;
    constexpr uint32_t       B = 16384;

    // Load files
    std::vector<std::string> files {};
    for (int i = 0; i < 21; i++)
        files.push_back("E:\\berserk-finny-data\\berserk9dev2.d9.finny" + std::to_string(i) + ".bin");

    BatchLoader  batch_loader {files, B, 1};

    // Input data (perspective)
    SparseInput  sparse_input_1 {I, B, 32};    // 32 max inputs
    SparseInput  sparse_input_2 {I, B, 32};

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

    constexpr int BATCHES_PER_EPOCH = 6100;
    logging::open("D:\\berserk\\nets\\experiments\\finny-trainer2\\log.txt");

    Timer t {};
    for (int epoch = 1; epoch <= 400; epoch++) {
        float epoch_loss = 0;
        t.tick();
        for (int batch = 0; batch < BATCHES_PER_EPOCH; batch++) {

            // get the next dataset (batch)
            auto* ds = batch_loader.next();
            // assign to the inputs and compute the target
            dense_berky::assign_inputs_batch(*ds,
                                                sparse_input_1,
                                                sparse_input_2,
                                                target,
                                                target_mask);
            // upload relevant data
            sparse_input_1.column_indices.gpu_upload();
            sparse_input_2.column_indices.gpu_upload();
            target.gpu_upload();
            target_mask.gpu_upload();

            // download the loss to display the loss of the iteration
            loss_function.loss.gpu_download();

            // measure time
            t.tock();
            std::stringstream ss {};
            ss << "    ep/ba = [" << std::setw(4) << epoch << "/" << std::setw(5) << batch << "]"
               << "    ba. loss  = " << std::setw(14) << loss_function.loss(0)
               << "    ep. loss  = " << std::setw(14) << epoch_loss / (batch + 1)
               << "    speed = " << std::setw(10)
               << std::round(1000.0f * (B * (batch + 1) / (float) t.duration()));
            std::cout << "\r" << ss.str() << std::flush;

            epoch_loss += loss_function.loss(0);
            // make sure to reset the loss here since the mse increments the loss in order to not have
            // to use memcpy (might change soon)
            loss_function.loss(0) = 0;
            loss_function.loss.gpu_upload();

            // feed forward
            network.batch(std::vector<SparseInput*> {&sparse_input_1, &sparse_input_2},
                          target,
                          target_mask);

            // update weights
            adam.apply(1);
        }
        std::cout << std::endl;
        logging::write("epoch          : " + std::to_string(epoch)
                       + "   train loss     : " + std::to_string(epoch_loss / BATCHES_PER_EPOCH));

        network.saveWeights("D:\\berserk\\nets\\experiments\\finny-trainer2\\weights-epoch" + std::to_string(epoch) + ".nn");
        quantitize("D:\\berserk\\nets\\experiments\\finny-trainer2\\nn-epoch" + std::to_string(epoch) + ".nnue", network);

        if (epoch % 300 == 0)
            adam.alpha *= 0.1f;
    }

    close();
}
