
//
// Created by Luecx on 10.11.2021.
//

#include "ReLU.h"
#include "../misc/logging.h"
#include "../operations/operations.h"

void ReLU::apply    (const SArray<float> &in,
                           SArray<float> &out, Mode mode) {

    if(mode == HOST){
        relu<HOST>(in, out);
    }else{
        relu<DEVICE>(in, out);
    }

}
void ReLU::backprop  (const SArray<float> &in,
                            SArray<float> &in_grd,
                      const SArray<float> &out,
                      const SArray<float> &out_grd, Mode mode) {
    if(mode == HOST){
        relu_bp<HOST>(in, in_grd, out, out_grd);
    }else{
        relu_bp<DEVICE>(in, in_grd, out, out_grd);
    }
}

void ReLU::logOverview() {logging::write("ReLU");}
