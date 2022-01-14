
//
// Created by Luecx on 10.11.2021.
//

#include "Linear.h"
#include <assert.h>
#include "../misc/logging.h"

void Linear::apply      (DenseMatrix &in, DenseMatrix &out, Mode mode){
    assert(out.values.size == out.values.size);
    if(mode == MODE_GPU){
        out.values.set<MODE_GPU>(out.values);
    }
    if(mode == MODE_CPU){
        out.values.set<MODE_CPU>(out.values);
    }
}
void Linear::backprop   (DenseMatrix &in, DenseMatrix &out, Mode mode){
    assert(out.values.size == out.values.size);
    if(mode == MODE_GPU){
        in.gradients.set<MODE_GPU>(out.gradients);
    }
    if(mode == MODE_CPU){
        in.gradients.set<MODE_CPU>(out.gradients);
    }
}

void Linear::logOverview() { logging::write("Linear"); }
