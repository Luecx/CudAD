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
#include "MLE.h"

// clang-format off
void MLE::apply(const SArray<float>& output,
                      SArray<float>& output_grad,
                const SArray<float>& target,
                const SArray<bool >& target_mask,
                Mode                 mode) {
    // clang-format on
    if (mode == HOST) {
        mle<HOST>(output, output_grad, target, target_mask, loss);
    } else {
        mle<DEVICE>(output, output_grad, target, target_mask, loss);
    }
}
void           MLE::logOverview() { logging::write("MLE"); }
SArray<float>& MLE::getLoss() { return loss; }
MLE::MLE() {
    loss.mallocGpu();
    loss.mallocCpu();
}
