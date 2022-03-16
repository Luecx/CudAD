
//
// Created by Luecx on 04.03.2022.
//
#include "MSE.h"
#include "../misc/logging.h"
void MSE::apply(const SArray<float>& output,
                      SArray<float>& output_grad,
                const SArray<float>& target,
                const SArray<bool >& target_mask,
                Mode                 mode) {
    if(mode == HOST){
        mse<HOST>(output, output_grad, target, target_mask, loss);
    }else{
        mse<DEVICE>(output, output_grad, target, target_mask, loss);
    }
}
void MSE::logOverview() {logging::write("MSE");}
SArray<float>& MSE::getLoss() { return loss; }
MSE::MSE() {
    loss.malloc_gpu();
    loss.malloc_cpu();
}