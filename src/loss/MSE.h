
//
// Created by Luecx on 18.02.2022.
//

#ifndef CUDAD_SRC_LOSS_MSE_H_
#define CUDAD_SRC_LOSS_MSE_H_

#include "Loss.h"
struct MSE : public Loss{
    SArray<float> loss{1};

    public:
    MSE();

    void apply(const SArray<float> &output,
                     SArray<float> &output_grad,
               const SArray<float> &target,
               const SArray<bool > &target_mask,
                     Mode          mode) override;
    void logOverview() override;
    SArray<float>& getLoss() override;
};


#endif    // CUDAD_SRC_LOSS_MSE_H_
