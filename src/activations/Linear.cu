
//
// Created by Luecx on 10.11.2021.
//

#include "Linear.h"
#include "../operations/operations.h"
#include "../misc/logging.h"


void Linear::apply      (const SArray<float> &in,
                               SArray<float> &out, Mode mode) {

    if(mode == HOST){
        add<HOST>(in, out, out, 1,0);
    }else{
        add<DEVICE>(in, out, out, 1,0);
    }

}
void Linear::backprop   (const SArray<float> &in,
                               SArray<float> &in_grd,
                         const SArray<float> &out,
                         const SArray<float> &out_grd, Mode mode) {
    if(mode == HOST){
        add<HOST>(out_grd, in_grd, in_grd, 1,0);
    }else{
        add<DEVICE>(out_grd, in_grd, in_grd, 1,0);
    }
}

void Linear::logOverview() { logging::write("Linear"); }
