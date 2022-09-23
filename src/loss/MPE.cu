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

#include "../misc/logging.h"
#include "MPE.h"

// clang-format off
void MPE::apply(const SArray<float>& output,
                      SArray<float>& output_grad,
                const SArray<float>& target,
                const SArray<bool >& target_mask,
                Mode                 mode) {
    // clang-format on
    if (mode == HOST) {
        mpe<HOST>(output, output_grad, target, target_mask, loss, m_power, m_avg_gradients);
    } else {
        mpe<DEVICE>(output, output_grad, target, target_mask, loss, m_power, m_avg_gradients);
    }
}
void MPE::logOverview() { logging::write("MPE(power=" + std::to_string(m_power) + ")"); }
// extract the loss
SArray<float>& MPE::getLoss() { return loss; }
// constructor
MPE::MPE(float power, bool avg_gradients) {
    m_power         = power;
    m_avg_gradients = avg_gradients;
    loss.mallocGpu();
    loss.mallocCpu();
}
