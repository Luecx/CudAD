
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
