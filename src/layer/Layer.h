//
// Created by Luecx on 23.02.2021.
//

#ifndef DIFFERENTIATION_LAYER_H
#define DIFFERENTIATION_LAYER_H

#include "../data/DenseMatrix.h"
#include "../data/SparseInput.h"
#include "../activations/Activation.h"
#include "../data/Tape.h"

#include <vector>

class ThreadData;

struct LayerInterface{
protected:
    int layerID = 0;
public:
//    virtual DenseMatrix* getBias   () = 0;
//    virtual DenseMatrix* getWeights() = 0;

    virtual int getOutputSize() = 0;
    virtual int getInputSize()  = 0;

    virtual Activation* getActivationFunction() = 0;
    virtual std::vector<Tape*> getTunableParameters() = 0;

    virtual void apply   (Tape &input, Tape& out) = 0;
    virtual void backprop(Tape &input, Tape& out) = 0;
    virtual void apply   (SparseInput &input, Tape& out) = 0;
    virtual void backprop(SparseInput &input, Tape& out) = 0;

    void assignID(int id){
        layerID = id;
    }
};

#endif //DIFFERENTIATION_LAYER_H
