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
#include "dataset/shuffle.h"
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
#include "misc/csv.h"

#include <filesystem>
#include <iostream>

const std::string data_path = "H:/Koivisto Resourcen/Training Data 7.9/";
std::string output = "../resources/runs/experiment_22/";

float validate(Network&     network,
                           DataSet&     data_set,
                           DenseMatrix& target,
                           SArray<bool>& target_mask,
                           SparseInput& i1,
                           SparseInput& i2) {

    int B = i1.n;

    // reset loss
    float prev_loss = network.getLossFunction()->getLoss().get(0);
    network.getLossFunction()->getLoss().get(0) = 0;
    network.getLossFunction()->getLoss().gpu_upload();

    int c = std::floor(data_set.positions.size() / B);
    for(int i = 0; i < c; i++){
        int id1 = i   * B;
        int id2 = id1 + B;
        DataSet temp{};
        temp.header.position_count = B;
        temp.positions.assign(&data_set.positions[id1],&data_set.positions[id2]);

        dense_berky::assign_inputs_batch(temp, i1, i2, target, target_mask);

        i1.column_indices.gpu_upload();
        i2.column_indices.gpu_upload();
        target.gpu_upload();
        target_mask.gpu_upload();

        network.feed(std::vector<SparseInput*> {&i1, &i2});

        network.getLossFunction()->apply(network.getOutput().values,
                                         network.getOutput().gradients,
                                         target,
                                         target_mask,
                                         DEVICE);
    }

    network.getLossFunction()->getLoss().gpu_download();
    float loss = network.getLossFunction()->getLoss().get(0);

    network.getLossFunction()->getLoss().get(0) = prev_loss;
    network.getLossFunction()->getLoss().gpu_upload();

    return loss / c;
}
int main() {
    init();

    // definitions
    constexpr uint32_t       I = 16 * 12 * 64;
    constexpr uint32_t       H = 512;
    constexpr uint32_t       H2= 16;
    constexpr uint32_t       O = 1;
    constexpr uint32_t       B = 16384;
    constexpr uint32_t     BPE = 6100;
    constexpr  int32_t       E = 600;

    // Load files
    std::vector<std::string> files {};
    for (int i = 0; i < 24; i++)
        files.push_back(data_path + "reshuffled_generated_" + std::to_string(i) + ".txt.bin");

    BatchLoader  batch_loader {files, B};

    // Input data (perspective)
    SparseInput  i0 {I, B, 32};    // 32 max inputs
    SparseInput  i1 {I, B, 32};

    DenseMatrix  target {O, B};
    SArray<bool> target_mask {O * B};
    target_mask.malloc_cpu();
    target_mask.malloc_gpu();

    // 1536 -> (2x512) -> 1
    DuplicateDenseLayer<I, H, Linear> l1 {};
    DenseLayer<H * 2, H2, ReLU>   l2 {};
    DenseLayer<H2, O, Linear>   l3 {};
//    l2.getTunableParameters()[0]->min_allowed_value = 0;
//    l2.getTunableParameters()[0]->max_allowed_value = 1;

    // Berserk sigmoid
//    dynamic_cast<Sigmoid*>(l3.getActivationFunction())->scalar = 2.5 / 400;

//    l1.lasso_regularization = (1.0 / 8388608.0);

    // stack layers to build network
    std::vector<LayerInterface*> layers {};
    layers.push_back(&l1);
    layers.push_back(&l2);
    layers.push_back(&l3);

    Network network {layers};

    // loss function
    MPE     loss_function {2.5, true};
    network.setLossFunction(&loss_function);

    // optimizer
    Adam adam {};
    adam.init(layers);
    adam.alpha = 0.0001;
    adam.beta1 = 0.95;
    adam.beta2 = 0.999;

    network.loadWeights(output + "networks/weights-epoch" + std::to_string(600) + ".nn");
    computeScalars(batch_loader, network, 8192, I);
//    test_fen(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w QKqk - 0 1", I);
//    quantitize(output + std::to_string(600) + ".net", network, 16, 256);

//    Timer t {};
//    CSVWriter csv_writer{output + "loss.csv"};
////    logging::open(output + "loss.csv");
//    for (int epoch = 1; epoch <= E; epoch++) {
//        float epoch_loss = 0;
//        t.tick();
//        for (int batch = 0; batch < BPE; batch++) {
//
//            // get the next dataset (batch)
//            auto* ds = batch_loader.next();
//            // assign to the inputs and compute the target
//            dense_relative::assign_inputs_batch(*ds, i0, i1, target, target_mask);
//            // upload relevant data
//            i0.column_indices.gpu_upload();
//            i1.column_indices.gpu_upload();
//            target.gpu_upload();
//            target_mask.gpu_upload();
//
//            // download the loss to display the loss of the iteration
//            loss_function.loss.gpu_download();
//
//            // measure time and print output
//            t.tock();
//            std::printf("\rep/ba = [%3d/%5d], ", epoch, batch + 1);
//            std::printf("batch_loss = [%1.8f], ", loss_function.loss(0));
//            std::printf("epoch_loss = [%1.8f], ", epoch_loss / (batch + 1));
//            std::printf("speed = [%9d pos/s], ", (int) std::round(1000.0f * B * (batch + 1) / t.duration()));
//            std::printf("time = [%3ds]", (int) t.duration() / 1000);
//            std::cout << std::flush;
//
//            epoch_loss += loss_function.loss(0);
//            // make sure to reset the loss here since the mse increments the loss in order to not have
//            // to use memcpy (might change soon)
//            loss_function.loss(0) = 0;
//            loss_function.loss.gpu_upload();
//
//            // feed forward
//            network.batch(std::vector<SparseInput*> {&i0, &i1}, target, target_mask);
//
//            // update weights
//            adam.apply(1);
//        }
//        std::cout << std::endl;
//        csv_writer.write({std::to_string(epoch),
//                          std::to_string(epoch_loss / BPE)});
//
//        network.saveWeights(output + "networks/weights-epoch" + std::to_string(epoch) + ".nn");
//
//        computeScalars(batch_loader, network, 128, I);
//        if (epoch % (29 * 10) == 0)
//            adam.alpha *= 0.1f;
//        if (epoch == 3){
//            adam.alpha = 0.01;
//        }
//    }

    close();
}
