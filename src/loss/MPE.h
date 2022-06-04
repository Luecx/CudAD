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
