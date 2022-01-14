
#include "ClippedReLU.h"
#include "../misc/logging.h"
#include "../operations/operations.h"

void ClippedReLU::apply    (const SArray<float> &in,
                                  SArray<float> &out, Mode mode) {

    if(mode == HOST){
        clipped_relu<HOST>(in, out, max);
    }else{
        clipped_relu<DEVICE>(in, out, max);
    }

}
void ClippedReLU::backprop  (const SArray<float> &in,
                                   SArray<float> &in_grd,
                             const SArray<float> &out,
                             const SArray<float> &out_grd, Mode mode) {
    if(mode == HOST){
        clipped_relu_bp<HOST>(in, in_grd, out, out_grd, max);
    }else{
        clipped_relu_bp<DEVICE>(in, in_grd, out, out_grd, max);
    }
}

void ClippedReLU::logOverview() {logging::write("ClippedReLU (" + std::to_string(max) + ")");}

