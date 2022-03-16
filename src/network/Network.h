//
// Created by Luecx on 19.01.2022.
//

#ifndef CUDAD_SRC_NETWORK_NETWORK_H_
#define CUDAD_SRC_NETWORK_NETWORK_H_

#include <utility>

#include "../layer/Layer.h"
#include "../operations/mse/mse.h"
#include "../loss/Loss.h"
class Network{

    private:

    std::vector<LayerInterface*> layers{};
    std::vector<Tape> output_tapes{};
    Loss* loss_function{};

    public:
    explicit Network(std::vector<LayerInterface*> layers) : layers(std::move(layers)) {}

    void createOutputTapes(int batch_size);

    void batch(const std::vector<SparseInput*> &inputs, const DenseMatrix& target, const SArray<bool> &target_mask);

    void batch(const std::vector<Tape*> &inputs, const DenseMatrix& target, const SArray<bool> &target_mask);

    void feed(const std::vector<SparseInput*> &inputs);

    void feed(const std::vector<Tape*> &inputs);

    Loss*                        getLossFunction() const;
    void                         setLossFunction(Loss* loss_function);

    void loadWeights(const std::string& file);

    void saveWeights(const std::string& file);

    Tape& getOutput();
    std::vector<LayerInterface*> getLayers();

    Tape& getOutput(int layer_id);
};

#endif    // CUDAD_SRC_NETWORK_NETWORK_H_
