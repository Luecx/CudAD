//
// Created by Luecx on 29.06.2022.
//

#ifndef CUDAD_SRC_NUMERICAL_FINITE_DIFFERENCE_H_
#define CUDAD_SRC_NUMERICAL_FINITE_DIFFERENCE_H_

#include "../network/Network.h"
#include "../optimizer/Adam.h"

/**
 * checks the backpropagation of the network
 * - Assumes the batch size of the network is set to 1
 * - Can take very long on large networks. Use this experimentally
 * @param network
 */
void finite_difference(Network& network, DenseMatrix& target, SArray<bool> &target_mask){
    // create a pseudo optimiser to clear the gradients
    Adam adam{};
    adam.init(network.getLayers());
    adam.lr = 0;

    network.uploadInputs();

    // go through each layer
    for(int i = network.getLayers().size() - 1; i >= 0; i--){
        LayerInterface* l = network.getLayers()[i];

        std::cout << "--------------- LAYER " << i << " ------------------" << std::endl;

        if(l->getTunableParameters().empty())
            continue;

        for(Tape* t:l->getTunableParameters()){
            std::cout << t->values << std::endl;
        }
        for(Tape* t:l->getTunableParameters()){


            for(int wgt_idx = 0; wgt_idx < t->values.size(); wgt_idx++){
                std::cout << "\rwgt idx: " << wgt_idx << std::flush;

                auto current_weight = t->values.get(wgt_idx);
                auto margin         = current_weight * 0.1f + 1.0f;
                auto top_weight     = current_weight + margin;
                auto bot_weight     = current_weight - margin;

                t->values.get(wgt_idx) = top_weight;
                t->values.gpuUpload();
                // feed and download loss
                network.batch(target, target_mask);
                network.getLossFunction()->getLoss().gpuDownload();
                auto top_loss = network.getLossFunction()->getLoss().get(0);
                network.getLossFunction()->getLoss().get(0) = 0;
                network.getLossFunction()->getLoss().gpuUpload();

                // clear gradients
                adam.apply(network.getBatchSize());

                t->values.get(wgt_idx) = bot_weight;
                t->values.gpuUpload();
                // feed and download loss
                network.batch(target, target_mask);
                network.getLossFunction()->getLoss().gpuDownload();
                // clear gradients
                adam.apply(network.getBatchSize());

                auto bot_loss = network.getLossFunction()->getLoss().get(0);
                network.getLossFunction()->getLoss().get(0) = 0;
                network.getLossFunction()->getLoss().gpuUpload();

                t->values.get(wgt_idx) = current_weight;
                t->values.gpuUpload();
                // feed and download loss and gradients
                network.batch(target, target_mask);
                network.getLossFunction()->getLoss().gpuDownload();
                t->gradients.gpuDownload();

                auto autodiff_gradient = t->gradients.get(wgt_idx);
                network.getLossFunction()->getLoss().get(0) = 0;
                network.getLossFunction()->getLoss().gpuUpload();

                // clear gradients
                adam.apply(network.getBatchSize());

                // compute numerical gradient
                auto numerical_gradient = (top_loss - bot_loss) / (2 * margin);

                // compute the difference between numerical and autodiff gradients
                auto absolute_difference = (autodiff_gradient - numerical_gradient);
                auto relative_difference = absolute_difference / std::max(1e-7f, std::abs(numerical_gradient));

                // if the absolute and relative difference is wrong,
                if(absolute_difference > 1e-4f && relative_difference > 0.03f){
                    std::cerr << "\n" << wgt_idx <<" " << autodiff_gradient << " " << numerical_gradient << " " << top_loss << " " << bot_loss << std::endl;
                    exit(-1);
                }else{
//                    std::cout << "\n" << wgt_idx <<" " << autodiff_gradient << " " << numerical_gradient << " " << top_loss << " " << bot_loss << std::endl;
                }
            }
            std::cout << std::endl;
        }

    }
}

#endif    // CUDAD_SRC_NUMERICAL_FINITE_DIFFERENCE_H_
