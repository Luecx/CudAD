

//
// Created by Luecx on 18.02.2022.
//

#include "MLE.h"
#include "../misc/logging.h"

void MLE::apply(const SArray<float>& output,
                      SArray<float>& output_grad,
                const SArray<float>& target,
                const SArray<bool >& target_mask,
                Mode                 mode) {
    if(mode == HOST){
        mle<HOST>(output, output_grad, target, target_mask, loss);
    }else{
        mle<DEVICE>(output, output_grad, target, target_mask, loss);
    }
}
void MLE::logOverview() {logging::write("MLE");}
SArray<float>& MLE::getLoss() { return loss; }
MLE::MLE() {
    loss.malloc_gpu();
    loss.malloc_cpu();
}
