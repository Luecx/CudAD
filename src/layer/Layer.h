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

#ifndef DIFFERENTIATION_LAYER_H
#define DIFFERENTIATION_LAYER_H

#include "../activations/Activation.h"
#include "../data/DenseMatrix.h"
#include "../data/SparseInput.h"
#include "../data/Tape.h"

#include <vector>

struct LayerInterface {
    protected:
    int layerID = 0;

    public:
    //    virtual DenseMatrix* getBias   () = 0;
    //    virtual DenseMatrix* getWeights() = 0;

    virtual uint32_t           getOutputSize()                                       = 0;
    virtual uint32_t           getInputSize()                                        = 0;

    virtual Activation*        getActivationFunction()                               = 0;
    virtual std::vector<Tape*> getTunableParameters()                                = 0;

    virtual void               apply(std::vector<Tape*> inputs, Tape& out)           = 0;
    virtual void               backprop(std::vector<Tape*> inputs, Tape& out)        = 0;
    virtual void               apply(std::vector<SparseInput*> inputs, Tape& out)    = 0;
    virtual void               backprop(std::vector<SparseInput*> inputs, Tape& out) = 0;

    void                       assignID(int id) { layerID = id; }
};

#endif    // DIFFERENTIATION_LAYER_H
