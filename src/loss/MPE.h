

//
// Created by Luecx on 04.03.2022.
//

#ifndef CUDAD_SRC_LOSS_MPE_CUH_
#define CUDAD_SRC_LOSS_MPE_CUH_

#include "Loss.h"

struct MPE : public Loss {
    // loss and the power to use as well as defining if gradients
    // when backproping shall be averaged. default to true
    SArray<float> loss {2};
    float         m_power         = 2;
    bool          m_avg_gradients = true;
    // constructor
    MPE(float power, bool average_gradients = true);
    // clang-format off
    void apply(const SArray<float> &output,
                     SArray<float> &output_grad,
               const SArray<float> &target,
               const SArray<bool > &target_mask,
               Mode                 mode) override;
    // clang-format on
    void           logOverview() override;
    SArray<float>& getLoss() override;
};

#endif    // CUDAD_SRC_LOSS_MPE_CUH_
