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

#ifndef CUDAD_SRC_LOSS_LOSS_H_
#define CUDAD_SRC_LOSS_LOSS_H_
#include "../data/SArray.h"
#include "../data/Mode.h"
#include "../operations/operations.h"

struct Loss {
    SArray<float> loss {2};

    // clang-format off
    virtual void apply   (const SArray<float> &output,
                                SArray<float> &output_grad,
                          const SArray<float> &target,
                          const SArray<bool > &target_mask,
                                Mode mode) = 0;
    // clang-format on
    virtual void           logOverview() = 0;
    virtual SArray<float>& getLoss()     = 0;
};

#endif    // CUDAD_SRC_LOSS_LOSS_H_
