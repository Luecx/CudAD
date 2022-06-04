
//
// Created by Luecx on 10.11.2021.
//

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
