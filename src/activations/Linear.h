
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_ACITVATIONS_LINEAR_H_
#define DIFFERENTIATION_SRC_ACITVATIONS_LINEAR_H_

#include "../data/SArray.h"
#include "../data/DenseMatrix.h"
#include "Activation.h"

struct Linear : Activation {

    void apply      (const SArray<float> &in,
                           SArray<float> &out, Mode mode) ;
    void backprop   (const SArray<float> &in,
                           SArray<float> &in_grd,
                     const SArray<float> &out,
                     const SArray<float> &out_grd, Mode mode) ;
    void logOverview() override;
};

#endif    // DIFFERENTIATION_SRC_ACITVATIONS_LINEAR_H_
