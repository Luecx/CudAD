
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_ACITVATIONS_CLIPPEDRELU_H_
#define DIFFERENTIATION_SRC_ACITVATIONS_CLIPPEDRELU_H_

#include "../data/DenseMatrix.h"
#include "Activation.h"

struct ClippedReLU : Activation{
    float max = 127;
    void apply      (DenseMatrix &in, DenseMatrix &out, Mode mode);
    void backprop   (DenseMatrix &in, DenseMatrix &out, Mode mode);
    void logOverview() override;
};


#endif    // DIFFERENTIATION_SRC_ACITVATIONS_CLIPPEDRELU_H_
