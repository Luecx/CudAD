
//
// Created by Luecx on 20.01.2022.
//

#ifndef CUDAD_SRC_QUANTITIZE_H_
#define CUDAD_SRC_QUANTITIZE_H_

#include <string>
#include "network/Network.h"
#include "position/position.h"
#include "position/fenparsing.h"
#include "mappings.h"

void test_fen(Network& network, const std::string& fen, uint32_t input_size){

    SparseInput sp1{input_size, 1, 32};
    SparseInput sp2{input_size, 1, 32};

    Position p = parseFen(fen);

    SArray<float> target{1};
    target.malloc_cpu();
    SArray<bool> target_mask{1};
    target_mask.malloc_cpu();

    dense_relative::assign_input(p, sp1,sp2, target,target_mask,0);
    sp1.column_indices.gpu_upload();
    sp2.column_indices.gpu_upload();


    network.feed(std::vector<SparseInput*>{&sp1, &sp2});
    network.getOutput().values.gpu_download();

//    network.getOutput(0).values.gpu_download();
//    network.getOutput(1).values.gpu_download();

//    std::cout << sp1 << " " << sp2 << std::endl;
//    std::cout << network.getOutput(0).values << std::endl;
//    std::cout << network.getOutput(1).values << std::endl;

    std::cout << "testing fen: " << fen << std::endl;
    std::cout << "eval: " << network.getOutput().values << std::endl;
}

template<typename type>
void writeMatrix(FILE* file, DenseMatrix& matrix, float scaling, bool column_major=false){
    SArray<type> data{matrix.size};
    data.malloc_cpu();

    uint32_t m = matrix.m;
    uint32_t n = matrix.n;

    int idx = 0;
    for(uint32_t i = 0; i < (column_major ? n:m); i++){
        for(uint32_t j = 0; j < (column_major ? m:n); j++){
            float original_value = column_major ? matrix(j,i) : matrix(i,j);
            if (std::is_integral_v<type>){
                data(idx++) = static_cast<type>(round(original_value * scaling));
            } else if (std::is_floating_point_v<type>){
                data(idx++) = static_cast<type>(original_value * scaling);
            }
        }
    }

    fwrite(data.cpu_values, sizeof(type), data.size, file);
}

template<typename wgt_type, typename bia_type>
void writeLayer(FILE* file, Tape* tunable_values, float wgt_scaling, float bia_scaling, bool column_major=false){
    uint32_t m = tunable_values->values.m;
    uint32_t n = tunable_values->values.n-1;
    DenseMatrix wgt{tunable_values->values, 0,0,m,n};
    DenseMatrix bia{tunable_values->values, 0,n,m,1};

    writeMatrix<wgt_type>(file, wgt, wgt_scaling, column_major);
    writeMatrix<bia_type>(file, bia, bia_scaling);
}

void quantitize(const std::string& path, Network& network, float scalar_1 = 1.0f, float scalar_2 = 1.0f) {
    FILE *f = fopen(path.c_str(), "wb");

    writeLayer<float, float>(f, network.getLayers()[0]->getTunableParameters()[0], scalar_1, scalar_1, true);
    writeLayer<float, float>(f, network.getLayers()[1]->getTunableParameters()[0], scalar_2, scalar_1 * scalar_2, false);

    fclose(f);
}

void computeScalars(BatchLoader& batch_loader, Network& network, int batches, uint32_t input_size){

    SparseInput sparse_input_1{input_size, (uint32_t) batch_loader.batch_size, 32};
    SparseInput sparse_input_2{input_size, (uint32_t) batch_loader.batch_size, 32};
    SArray<float> target{(uint32_t) batch_loader.batch_size};
    target.malloc_cpu();
    SArray<bool> target_mask{(uint32_t) batch_loader.batch_size};
    target_mask.malloc_cpu();

    std::vector<SArray<float>> maximum{};
    std::vector<SArray<float>> minimum{};
    std::vector<float> maximum_wgt{};
    std::vector<float> minimum_wgt{};
    for(LayerInterface* layer_interface:network.getLayers()){
        maximum.emplace_back(layer_interface->getOutputSize());
        maximum[maximum.size()-1].malloc_cpu();
        minimum.emplace_back(layer_interface->getOutputSize());
        minimum[minimum.size()-1].malloc_cpu();

        maximum_wgt.emplace_back(layer_interface->getTunableParameters()[0]->values.max());
        minimum_wgt.emplace_back(layer_interface->getTunableParameters()[0]->values.min());
    }

    for (int batch = 0; batch < batches; batch++){
        // get the next dataset (batch)
        auto* ds = batch_loader.next();
        // assign to the inputs and compute the target
        dense_relative::assign_inputs_batch(*ds, sparse_input_1, sparse_input_2, target,target_mask);
        // upload relevant data
        sparse_input_1.column_indices .gpu_upload();
        sparse_input_2.column_indices .gpu_upload();
        target                        .gpu_upload();
        target_mask                   .gpu_upload();
        network.feed(std::vector<SparseInput*>{&sparse_input_1, &sparse_input_2});

        std::cout << "\rProcessing batch: " << (batch + 1) << "/" << batches << std::flush;

        // iterate over the layers
        for(int i = 0; i < maximum.size(); i++){

            network.getOutput(i).values.gpu_download();

            for(int j = 0; j < network.getLayers()[i]->getOutputSize(); j++){

                maximum[i](j) = std::max(maximum[i](j), network.getOutput(i).values(j));
                minimum[i](j) = std::min(minimum[i](j), network.getOutput(i).values(j));
            }
        }
    }
    std::cout << std::endl;

    for(int i = 0; i < maximum.size(); i++){
//        std::cout << minimum[i] << "\n" << maximum[i] << std::endl;
        std::cout << "layer  : " << i << std::endl;
        std::cout << "min    : " << std::left << std::setw(10) << minimum[i].min()
                  << "max    : " << std::left << std::setw(10) << maximum[i].max()
                  << "min wgt: " << std::left << std::setw(10) << minimum_wgt[i]
                  << "max wgt: " << std::left << std::setw(10) << maximum_wgt[i] << std::endl;
    }
}

#endif    // CUDAD_SRC_QUANTITIZE_H_
