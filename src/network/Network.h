//
// Created by Luecx on 19.01.2022.
//

#ifndef CUDAD_SRC_NETWORK_NETWORK_H_
#define CUDAD_SRC_NETWORK_NETWORK_H_

#include <utility>

#include "../layer/Layer.h"
#include "../operations/mse/mse.h"
class Network{

    private:

    std::vector<LayerInterface*> layers{};
    std::vector<Tape> output_tapes{};

    public:
    explicit Network(std::vector<LayerInterface*> layers) : layers(std::move(layers)) {}

    void createOutputTapes(int batch_size);

    void batch(const std::vector<SparseInput*> &inputs, const DenseMatrix& target, SArray<float>& loss);

    void feed(const std::vector<SparseInput*> &inputs);

    void loadWeights(const std::string& file);

    void saveWeights(const std::string& file);

    Tape& getOutput();
    std::vector<LayerInterface*> getLayers();

    Tape& getOutput(int layer_id);
};

#endif    // CUDAD_SRC_NETWORK_NETWORK_H_
