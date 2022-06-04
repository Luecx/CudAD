
//
// Created by Luecx on 18.02.2022.
//

#ifndef CUDAD_SRC_LOSS_MLE_H_
#define CUDAD_SRC_LOSS_MLE_H_

#include "Loss.h"
struct MLE : public Loss {
    SArray<float> loss {2};
    MLE();
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

#endif    // CUDAD_SRC_LOSS_MLE_H_
