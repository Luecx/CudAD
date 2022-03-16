
//
// Created by Luecx on 18.02.2022.
//

#ifndef CUDAD_SRC_LOSS_LOSS_H_
#define CUDAD_SRC_LOSS_LOSS_H_
#include "../data/mode.h"
#include "../data/SArray.h"
#include "../operations/operations.h"

struct Loss{


    virtual void apply   (const SArray<float> &output,
                                SArray<float> &output_grad,
                          const SArray<float> &target,
                          const SArray<bool > &target_mask,
                                Mode mode) = 0;
    virtual void logOverview() = 0;
    virtual SArray<float>& getLoss() = 0;
};

#endif    // CUDAD_SRC_LOSS_LOSS_H_
