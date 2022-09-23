/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUDAD_SRC_TRAINER_H_
#define CUDAD_SRC_TRAINER_H_

#include "data/DenseMatrix.h"
#include "data/SArray.h"
#include "data/SparseInput.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "loss/Loss.h"
#include "misc/csv.h"
#include "misc/timer.h"
#include "network/Network.h"
#include "optimizer/Optimiser.h"
#include "quantitize.h"

#include <tuple>

using namespace std;

template<class Arch, int Epochs = 1000, int BatchSize = 16384, int SamplesPerEpoch = 100000000>
class Trainer {
    static constexpr int MaxInputs       = 32;
    static constexpr int BatchesPerEpoch = SamplesPerEpoch / BatchSize;

    using LayerList = std::vector<LayerInterface*>;

    public:
    DenseMatrix                     target {Arch::Outputs, BatchSize};
    SArray<bool>                    target_mask {Arch::Outputs * BatchSize};
    Network*                        network;
    Loss*                           loss_f;
    Optimiser*                      optim;

    Trainer() {
        std::tuple<LayerList, LayerList> layers = Arch::get_layers();
        LayerList inputs_layers = std::get<0>(layers);
        LayerList hidden_layers = std::get<1>(layers);

        this->optim                    = Arch::get_optimiser();
        this->optim->init(hidden_layers);

        this->loss_f = Arch::get_loss_function();

        network      = new Network(inputs_layers, hidden_layers);
        network->setLossFunction(this->loss_f);
        network->setBatchSize(BatchSize);

        target_mask.mallocCpu();
        target_mask.mallocGpu();
    }

    void fit(vector<string> files, vector<string> validation_files, string output) {
        BatchLoader training_data {files, BatchSize};
        training_data.start();

        DataSet validation_data {};
        for (size_t i = 0; i < validation_files.size(); i++){

            DataSet ds = read<BINARY>(validation_files.at(i), 3e6);
            validation_data.addData(ds);
        }

        CSVWriter log {output + "loss.csv"};
        log.write({"epoch", "training_loss", "validation_loss"});

        Timer t {};
        for (int epoch = 1; epoch <= Epochs; epoch++) {
            t.tick();

            float epoch_loss      = train(epoch, &t, &training_data);
            float validation_loss = validate(&validation_data);

            t.tock();


            printf("\rep/ba = [%3d/%5d], ", epoch, BatchesPerEpoch);
            printf("valid_loss = [%1.8f], ", validation_loss);
            printf("epoch_loss = [%1.8f], ", epoch_loss / BatchesPerEpoch);
            printf("speed = [%9d pos/s], ",
                   (int) round(1000.0f
                               * (BatchSize * BatchesPerEpoch + validation_data.header.position_count)
                               / t.duration()));
            printf("time = [%3ds]", (int) t.duration() / 1000);
            cout << endl;

            log.write({std::to_string(epoch),
                       std::to_string(epoch_loss / BatchesPerEpoch),
                       std::to_string(validation_loss)});

            if (epoch % 10 == 0) {
//                quantitize_shallow(output + "nn-epoch" + std::to_string(epoch) + ".nnue", *network);
                network->saveWeights(output + "weights-epoch" + std::to_string(epoch) + ".nnue");
            }

            if (epoch % optim->schedule.step == 0)
                optim->lr *= optim->schedule.gamma;
        }
    }

    float train(int epoch, Timer* timer, BatchLoader* batch_loader) {


        float     epoch_loss    = 0.0;
        long long prev_duration = 0;

        for (int batch = 1; batch <= BatchesPerEpoch; batch++) {
            auto* ds = batch_loader->next();

            Arch::assign_inputs_batch(*ds, *network, target, target_mask);
            network->uploadInputs();

            target.gpuUpload();
            target_mask.gpuUpload();
            loss_f->loss.gpuDownload();

            // measure time and print output
            timer->tock();
            if (batch == BatchesPerEpoch || timer->duration() - prev_duration > 1000) {
                prev_duration = timer->duration();

                printf("\rep/ba = [%3d/%5d], ", epoch, batch + 1);
                printf("batch_loss = [%1.8f], ", loss_f->loss(0));
                printf("epoch_loss = [%1.8f], ", epoch_loss / (batch + 1));
                printf("speed = [%9d pos/s], ",
                       (int) round(1000.0f * BatchSize * (batch + 1) / timer->duration()));
                printf("time = [%3ds]", (int) timer->duration() / 1000);
                cout << flush;
            }

            epoch_loss += loss_f->loss(0);
            loss_f->loss(0) = 0;
            loss_f->loss.gpuUpload();
            network->batch(target,
                           target_mask);
            optim->apply(1);
        }

        return epoch_loss;
    }

    float validate(DataSet* validation_data) {
        float prev_loss = loss_f->loss(0);
        loss_f->loss(0) = 0;
        loss_f->loss.gpuUpload();

        float total_loss_sum = 0;

        int c = floor(validation_data->positions.size() / BatchSize);
        for (int i = 0; i < c; i++) {
            int     id1 = i * BatchSize;
            int     id2 = id1 + BatchSize;

            DataSet temp {};
            temp.header.position_count = BatchSize;
            temp.positions.assign(&validation_data->positions[id1], &validation_data->positions[id2]);

            Arch::assign_inputs_batch(temp, *network, target, target_mask);
            network->uploadInputs();

            target.gpuUpload();
            target_mask.gpuUpload();

            network->feed();

            loss_f->apply(network->getOutput().values,
                          network->getOutput().gradients,
                          target,
                          target_mask,
                          DEVICE);

            // reset loss to avoid loss of accuracy
            loss_f->loss.gpuDownload();
            total_loss_sum += loss_f->loss.get(0);
            loss_f->loss.get(0) = 0;
            loss_f->loss.gpuUpload();
        }


        loss_f->loss.gpuDownload();

        loss_f->loss(0)       = prev_loss;
        loss_f->loss.gpuUpload();

        return total_loss_sum / c;
    }
};

#endif