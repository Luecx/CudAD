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

#ifndef CUDAD_SRC_NETWORK_NETWORK_H_
#define CUDAD_SRC_NETWORK_NETWORK_H_

#include "../layer/Layer.h"
#include "../loss/Loss.h"
#include "../operations/mse/mse.h"

#include <fstream>
#include <utility>
#include <vector>

class Network {

    private:
    std::vector<LayerInterface*> inputs {};
    std::vector<LayerInterface*> layers {};
    Loss*                        loss_function {};

    int                          batch_size = -1;

    public:
    explicit Network(std::vector<LayerInterface*> inputs, std::vector<LayerInterface*> layers)
        : inputs {inputs}, layers {layers} {

                           };

    void feed() {
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->apply();
        }
    }
    void backprop() {
        for (int i = layers.size() - 1; i >= 0; i--) {
            layers[i]->backprop();
        }
    }
    void batch(DenseMatrix& target, SArray<bool>& target_mask) {
        ASSERT(target.isAllocated<DEVICE>());
        ASSERT(target_mask.isAllocated<DEVICE>());
        ASSERT(loss_function);
        feed();
        loss_function->apply(getOutput().values, getOutput().gradients, target, target_mask, DEVICE);
        backprop();
    }

    void loadWeights(const std::string& file) {
        FILE* f = fopen(file.c_str(), "rb");

        // figure out how many entries we will store
        uint64_t count = 0;
        for (LayerInterface* l : layers) {
            for (Tape* t : l->getTunableParameters()) {
                count += t->values.size();
            }
        }

        uint64_t fileCount = 0;
        fread(&fileCount, sizeof(uint64_t), 1, f);
        ASSERT(count == fileCount);

        for (LayerInterface* l : layers) {
            for (Tape* t : l->getTunableParameters()) {
                fread(t->values.address<HOST>(), sizeof(float), t->values.size(), f);
                t->values.gpuUpload();
            }
        }
        fclose(f);
    };
    void saveWeights(const std::string& file) {
        FILE* f = fopen(file.c_str(), "wb");

        // figure out how many entries we will store
        uint64_t count = 0;
        for (LayerInterface* l : layers) {
            for (Tape* t : l->getTunableParameters()) {
                count += t->values.size();
            }
        }

        fwrite(&count, sizeof(uint64_t), 1, f);
        for (LayerInterface* l : layers) {
            for (Tape* t : l->getTunableParameters()) {
                t->values.gpuDownload();
                fwrite(t->values.address<HOST>(), sizeof(float), t->values.size(), f);
            }
        }
        fclose(f);
    }

    void setBatchSize(int batch_size) {
        if (batch_size != this->batch_size) {
            this->batch_size = batch_size;
            for (LayerInterface* l : inputs) {
                l->createOutput(batch_size);
            }
            for (LayerInterface* l : layers) {
                l->createOutput(batch_size);
            }
        }
    }

    int getBatchSize(){
        return batch_size;
    }

    void uploadInputs(){
        for(LayerInterface* l:inputs){
            if(l->isSparse()){
                l->sparse_data.column_indices.gpuUpload();
            }else{
                l->dense_data.values.gpuUpload();
            }
        }
    }

    Tape&                        getOutput() { return getOutput(layers.size() - 1); };
    std::vector<LayerInterface*> getInputs() { return inputs; }
    std::vector<LayerInterface*> getLayers() { return layers; }
    Tape&                        getOutput(int layer_id) { return layers[layer_id]->getDenseData(); };

    Loss*                        getLossFunction() const { return loss_function; };
    void setLossFunction(Loss* loss_function) { this->loss_function = loss_function; }
};

#endif    // CUDAD_SRC_NETWORK_NETWORK_H_
