/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef DIFFERENTIATION_SRC_ACITVATIONS_LINEAR_H_
#define DIFFERENTIATION_SRC_ACITVATIONS_LINEAR_H_

#include "../data/DenseMatrix.h"
#include "../data/SArray.h"
#include "Activation.h"

struct Linear : Activation {
    // clang-format off

    void apply      (const SArray<float> &in,
                           SArray<float> &out, Mode mode) ;
    void backprop   (const SArray<float> &in,
                           SArray<float> &in_grd,
                     const SArray<float> &out,
                     const SArray<float> &out_grd, Mode mode) ;
    // clang-format on
    void logOverview() override;
};

#endif    // DIFFERENTIATION_SRC_ACITVATIONS_LINEAR_H_
