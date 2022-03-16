

//
// Created by Luecx on 04.03.2022.
//

#ifndef CUDAD_SRC_LOSS_MPE_CUH_
#define CUDAD_SRC_LOSS_MPE_CUH_

#include "Loss.h"

struct MPE : public Loss{
    SArray<float> loss{2};
    float m_power = 2;
    MPE(float power);
    void apply(const SArray<float> &output,
                     SArray<float> &output_grad,
               const SArray<float> &target,
               const SArray<bool > &target_mask,
               Mode                 mode) override;
    void logOverview() override;
    SArray<float>& getLoss() override;
};

#endif    // CUDAD_SRC_LOSS_MPE_CUH_
