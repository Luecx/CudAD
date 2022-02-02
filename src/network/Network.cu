
//
// Created by Luecx on 19.01.2022.
//
#include "Network.h"

void Network::loadWeights(const std::string& file) {
    FILE *f = fopen(file.c_str(), "rb");

    // figure out how many entries we will store
    uint64_t count = 0;
    for(LayerInterface* l:layers){
        for(Tape* t:l->getTunableParameters()){
            count += t->values.size;
        }
    }

    uint64_t fileCount = 0;
    fread(&fileCount, sizeof(uint64_t), 1, f);
    ASSERT(count == fileCount);

    for(LayerInterface* l:layers){
        for(Tape* t:l->getTunableParameters()){
            fread(t->values.cpu_values, sizeof(float), t->values.size, f);
            t->values.gpu_upload();
        }
    }
    fclose(f);
}
void Network::saveWeights(const std::string& file) {
    FILE *f = fopen(file.c_str(), "wb");

    // figure out how many entries we will store
    uint64_t count = 0;
    for(LayerInterface* l:layers){
        for(Tape* t:l->getTunableParameters()){
            count += t->values.size;
        }
    }

    fwrite(&count, sizeof(uint64_t), 1, f);
    for(LayerInterface* l:layers){
        for(Tape* t:l->getTunableParameters()){
            t->values.gpu_download();
            fwrite(t->values.cpu_values, sizeof(float), t->values.size, f);
        }
    }
    fclose(f);
}
void Network::batch(const std::vector<SparseInput*>& inputs,
                    const DenseMatrix&               target,
                    SArray<float>&                   loss) {
    createOutputTapes(inputs[0]->n);
    layers[0]->apply(inputs, output_tapes[0]);
    for(int i = 1; i < layers.size(); i++){
        layers[i]->apply({&output_tapes[i-1]}, output_tapes[i]);
    }


    mse<DEVICE>(output_tapes[output_tapes.size()-1].values,
                output_tapes[output_tapes.size()-1].gradients,
                target,
                loss);

    for(int i = layers.size() - 1; i >= 1; i--){
        layers[i]->backprop({&output_tapes[i-1]}, output_tapes[i]);
    }
    layers[0]->backprop(inputs, output_tapes[0]);
}

void Network::feed(const std::vector<SparseInput*> &inputs){
    createOutputTapes(inputs[0]->n);
    layers[0]->apply(inputs, output_tapes[0]);
    for(int i = 1; i < layers.size(); i++){
        layers[i]->apply({&output_tapes[i-1]}, output_tapes[i]);
    }
}


void Network::createOutputTapes(int batch_size) {
    // if there are tapes already with the correct size, dont create new tapes
    if(!output_tapes.empty() && output_tapes[0].values.n == batch_size) return;
    // clear the tapes
    output_tapes.clear();
    // create a mew tape
    for(int i = 0; i < layers.size(); i++){
        output_tapes.emplace_back((uint32_t)layers[i]->getOutputSize(), // output of layer
                                  (uint32_t)batch_size);                // batch size
    }
}
Tape& Network::getOutput() {
    return output_tapes[output_tapes.size()-1];
}
std::vector<LayerInterface*> Network::getLayers() { return layers; }
Tape&                        Network::getOutput(int layer_id) {
    return output_tapes[layer_id];
}
