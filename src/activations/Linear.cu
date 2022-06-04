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
#include "Linear.h"

// clang-format off
void Linear::apply      (const SArray<float> &in,
                               SArray<float> &out, Mode mode) {
    // clang-format on

    if (mode == HOST) {
        add<HOST>(in, out, out, 1, 0);
    } else {
        add<DEVICE>(in, out, out, 1, 0);
    }
}

// clang-format off
void Linear::backprop   (const SArray<float> &in,
                               SArray<float> &in_grd,
                         const SArray<float> &out,
                         const SArray<float> &out_grd, Mode mode) {
    // clang-format on

    if (mode == HOST) {
        add<HOST>(out_grd, in_grd, in_grd, 1, 0);
    } else {
        add<DEVICE>(out_grd, in_grd, in_grd, 1, 0);
    }
}

void Linear::logOverview() { logging::write("Linear"); }
