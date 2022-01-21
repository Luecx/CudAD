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


struct LayerInterface{
protected:
    int layerID = 0;
public:
//    virtual DenseMatrix* getBias   () = 0;
//    virtual DenseMatrix* getWeights() = 0;

    virtual uint32_t getOutputSize() = 0;
    virtual uint32_t getInputSize()  = 0;

    virtual Activation* getActivationFunction() = 0;
    virtual std::vector<Tape*> getTunableParameters() = 0;

    virtual void apply   (std::vector<Tape*> inputs, Tape& out) = 0;
    virtual void backprop(std::vector<Tape*> inputs, Tape& out) = 0;
    virtual void apply   (std::vector<SparseInput*> inputs, Tape& out) = 0;
    virtual void backprop(std::vector<SparseInput*> inputs, Tape& out) = 0;

    void assignID(int id){
        layerID = id;
    }
};

#endif //DIFFERENTIATION_LAYER_H
