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
#include "ClippedReLU.h"

// clang-format off
void ClippedReLU::apply    (const SArray<float> &in,
                                  SArray<float> &out, Mode mode) {
    // clang-format on
    if (mode == HOST) {
        clipped_relu<HOST>(in, out, max);
    } else {
        clipped_relu<DEVICE>(in, out, max);
    }
}
// clang-format off
void ClippedReLU::backprop  (const SArray<float> &in,
                                   SArray<float> &in_grd,
                             const SArray<float> &out,
                             const SArray<float> &out_grd, Mode mode) {
    // clang-format on
    if (mode == HOST) {
        clipped_relu_bp<HOST>(in, in_grd, out, out_grd, max);
    } else {
        clipped_relu_bp<DEVICE>(in, in_grd, out, out_grd, max);
    }
}

void ClippedReLU::logOverview() { logging::write("ClippedReLU (" + std::to_string(max) + ")"); }
