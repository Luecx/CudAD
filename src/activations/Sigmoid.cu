
//
// Created by Luecx on 10.11.2021.
//

#include "Sigmoid.h"
#include "../misc/logging.h"
#include "../operations/operations.h"
void Sigmoid::apply    (const SArray<float> &in,
                              SArray<float> &out, Mode mode) {

    if(mode == HOST){
        sigmoid<HOST>(in, out, scalar);
    }else{
        sigmoid<DEVICE>(in, out, scalar);
    }

}
void Sigmoid::backprop  (const SArray<float> &in,
                               SArray<float> &in_grd,
                         const SArray<float> &out,
                         const SArray<float> &out_grd, Mode mode) {
    if(mode == HOST){
        sigmoid_bp<HOST>(in, in_grd, out, out_grd, scalar);
    }else{
        sigmoid_bp<DEVICE>(in, in_grd, out, out_grd, scalar);
    }
}

void Sigmoid::logOverview() { logging::write("Sigmoid (" + std::to_string(scalar) + ")"); }

#include "Sigmoid.h"
