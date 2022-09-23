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

#ifndef CUDAD_SRC_QUANTITIZE_H_
#define CUDAD_SRC_QUANTITIZE_H_

#include "network/Network.h"
#include "position/fenparsing.h"
#include "position/position.h"

#include <string>

template<class Arch>
void test_fen(Network& network, const std::string& fen) {

    Position      p = parseFen(fen);

    network.setBatchSize(1);

    // TODO
    SparseInput& in1 = network.getInputs()[0]->sparse_data;
    SparseInput& in2 = network.getInputs()[1]->sparse_data;
//    DenseMatrix& in3 = network.getInputs()[2]->dense_data.values;
    in1.clear();
    in2.clear();

    SArray<float> target {Arch::Outputs};
    target.mallocCpu();
    SArray<bool> target_mask {Arch::Outputs};
    target_mask.mallocCpu();

    Arch::assign_input(p, in1, in2, target, target_mask, 0);

    network.uploadInputs();
    network.feed();

    std::cout << "==================================================================================\n";
    std::cout << "testing fen: " << fen << std::endl;
    int idx = 0;
    for(LayerInterface* l:network.getLayers()){
        l->getDenseData().values.gpuDownload();

        std::cout << "LAYER " << (idx ++) << std::endl;
        for(int i = 0; i < std::min(16u, l->getDenseData().values.size()); i++){
            std::cout << std::setw(10) << l->getDenseData().values(i);
        }
        if(l->getDenseData().values.size() > 16){
            std::cout << " ......... " << l->getDenseData().values(l->getDenseData().values.size()-1);
        }
        std::cout << "\n";
    }
}

template<typename type>
void writeMatrix(FILE* file, DenseMatrix& matrix, float scaling, bool column_major = false) {
    SArray<type> data {matrix.size()};
    data.mallocCpu();

    uint32_t m   = matrix.m;
    uint32_t n   = matrix.n;

    int      idx = 0;
    for (uint32_t i = 0; i < (column_major ? n : m); i++) {
        for (uint32_t j = 0; j < (column_major ? m : n); j++) {
            float original_value = column_major ? matrix(j, i) : matrix(i, j);

            if (std::is_integral_v<type>) {
                data(idx++) = static_cast<type>(round(original_value * scaling));
            } else if (std::is_floating_point_v<type>) {
                data(idx++) = static_cast<type>(original_value * scaling);
            }
        }
    }

    fwrite(data.cpuAddress(), sizeof(type), data.size(), file);
}

FILE* openFile(const std::string& path){
    FILE* f          = fopen(path.c_str(), "wb");
    return f;
}

void closeFile(FILE* f){
    fclose(f);
}

template<typename T1, typename T2>
void writeLayer(FILE* file, Network& network, int layer_id, int s1, int s2){
    auto  l0         = network.getLayers()[layer_id];
    auto  l0_params  = l0->getTunableParameters();
    auto  l0_weights = l0_params[0]->values;
    auto  l0_biases  = l0_params[1]->values;

    l0_weights.gpuDownload(), l0_biases.gpuDownload();

    writeMatrix<T1>(file, l0_weights, s1, layer_id == 0);
    writeMatrix<T2>(file, l0_biases, s2);
}

void quantitize_shallow(const std::string& path,
                        Network&           network,
                        float              scalar_1 = 16,
                        float              scalar_2 = 512) {
    FILE* f          = fopen(path.c_str(), "wb");

    auto  l0         = network.getLayers()[0];
    auto  l0_params  = l0->getTunableParameters();
    auto  l0_weights = l0_params[0]->values;
    auto  l0_biases  = l0_params[1]->values;

    l0_weights.gpuDownload(), l0_biases.gpuDownload();
    writeMatrix<int16_t>(f, l0_weights, scalar_1, true);
    writeMatrix<int16_t>(f, l0_biases, scalar_1);

    auto l1         = network.getLayers()[1];
    auto l1_params  = l1->getTunableParameters();
    auto l1_weights = l1_params[0]->values;
    auto l1_biases  = l1_params[1]->values;

    l1_weights.gpuDownload(), l1_biases.gpuDownload();
    writeMatrix<int16_t>(f, l1_weights, scalar_2);
    writeMatrix<int32_t>(f, l1_biases, scalar_1 * scalar_2);

    fclose(f);
}

template<class Arch>
void computeScalars(BatchLoader& batch_loader, Network& network, int batches) {

    network.setBatchSize(batch_loader.batch_size);

    SparseInput   sparse_input_1 {Arch::Inputs, (uint32_t) batch_loader.batch_size, 32};
    SparseInput   sparse_input_2 {Arch::Inputs, (uint32_t) batch_loader.batch_size, 32};
    SArray<float> target {(uint32_t) batch_loader.batch_size * Arch::Outputs};
    target.mallocCpu();
    SArray<bool> target_mask {(uint32_t) batch_loader.batch_size * Arch::Outputs};
    target_mask.mallocCpu();

    std::vector<SArray<float>> maximum {};
    std::vector<SArray<float>> minimum {};
    std::vector<float>         maximum_wgt {};
    std::vector<float>         minimum_wgt {};
    for (LayerInterface* layer_interface : network.getLayers()) {
        maximum.emplace_back(layer_interface->getOutputSize());
        maximum[maximum.size() - 1].mallocCpu();
        minimum.emplace_back(layer_interface->getOutputSize());
        minimum[minimum.size() - 1].mallocCpu();

        if(layer_interface->getTunableParameters().size()){
            maximum_wgt.emplace_back(layer_interface->getTunableParameters()[0]->values.max());
            minimum_wgt.emplace_back(layer_interface->getTunableParameters()[0]->values.min());
        }else{
            maximum_wgt.emplace_back(0);
            minimum_wgt.emplace_back(0);
        }

    }

    for (int batch = 0; batch < batches; batch++) {
        // get the next dataset (batch)
        auto* ds = batch_loader.next();
        // assign to the inputs and compute the target
        Arch::assign_inputs_batch(*ds, network, target, target_mask);
        // upload relevant data
        network.uploadInputs();
        target.gpuUpload();
        target_mask.gpuUpload();
        network.feed();

        std::cout << "\rProcessing batch: " << (batch + 1) << "/" << batches << std::flush;

        // iterate over the layers
        for (int i = 0; i < maximum.size(); i++) {

            network.getOutput(i).values.gpuDownload();

            for (int j = 0; j < network.getLayers()[i]->getOutputSize(); j++) {

                maximum[i](j) = std::max(maximum[i](j), network.getOutput(i).values(j));
                minimum[i](j) = std::min(minimum[i](j), network.getOutput(i).values(j));
            }
        }
    }
    std::cout << std::endl;

    for (int i = 0; i < maximum.size(); i++) {
//        std::cout << minimum[i] << "\n" << maximum[i] << std::endl;

        int died = 0;
        for (int j = 0; j < minimum[i].size(); j++) {
            if (abs(maximum[i].get(j) - minimum[i].get(j)) < 1e-8) {
                died++;
            }
        }

        std::cout << "layer  : " << i << std::endl;
        std::cout << "min    : " << std::left << std::setw(10) << minimum[i].min()
                  << "max    : " << std::left << std::setw(10) << maximum[i].max()
                  << "min wgt: " << std::left << std::setw(10) << minimum_wgt[i]
                  << "max wgt: " << std::left << std::setw(10) << maximum_wgt[i]
                  << "died   : " << std::left << std::setw(10) << died * 100 / minimum[i].size() << " %"
                  << std::endl;
    }
}

#endif    // CUDAD_SRC_QUANTITIZE_H_
