
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_ACITVATIONS_ACTIVATION_H_
#define DIFFERENTIATION_SRC_ACITVATIONS_ACTIVATION_H_

#include "../data/DenseMatrix.h"
#include "../data/SArray.h"
struct Activation {
    // clang-format off
    virtual void apply      (const SArray<float> &in,
                                   SArray<float> &out, Mode mode) = 0;

    virtual void backprop   (const SArray<float> &in,
                                   SArray<float> &in_grd,
                             const SArray<float> &out,
                             const SArray<float> &out_grd, Mode mode) = 0;

    // clang-format on
    virtual void logOverview() = 0;
};

#endif    // DIFFERENTIATION_SRC_ACITVATIONS_ACTIVATION_H_
