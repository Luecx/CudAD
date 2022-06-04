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
#include "../operations/operations.h"
#include "ReLU.h"

// clang-format off
void ReLU::apply    (const SArray<float> &in,
                           SArray<float> &out, Mode mode) {
    // clang-format on

    if (mode == HOST) {
        relu<HOST>(in, out);
    } else {
        relu<DEVICE>(in, out);
    }
}

// clang-format off
void ReLU::backprop  (const SArray<float> &in,
                            SArray<float> &in_grd,
                      const SArray<float> &out,
                      const SArray<float> &out_grd, Mode mode) {
    // clang-format on
    if (mode == HOST) {
        relu_bp<HOST>(in, in_grd, out, out_grd);
    } else {
        relu_bp<DEVICE>(in, in_grd, out, out_grd);
    }
}

void ReLU::logOverview() { logging::write("ReLU"); }
