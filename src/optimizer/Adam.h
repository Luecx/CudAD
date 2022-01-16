//
// Created by Luecx on 16.01.2022.
//

#ifndef CUDAD_SRC_OPTIMIZER_ADAM_H_
#define CUDAD_SRC_OPTIMIZER_ADAM_H_

#include "Optimiser.h"
#include "../operations/adam/adam.h"

struct Adam : Optimiser {
    private:
    std::vector<SArray<float>> first_moments{};
    std::vector<SArray<float>> second_moments{};

    public:
    double alpha = 0.01;
    double beta1 = 0.99;
    double beta2 = 0.999;
    double eps   = 1e-8;

    virtual void createBuffers() {
        for(Tape* t:tunable_values){
            first_moments .push_back(SArray<float>{t->values.size});
            second_moments.push_back(SArray<float>{t->values.size});
            first_moments [first_moments.size()-1].malloc_gpu();
            second_moments[first_moments.size()-1].malloc_gpu();
        }
    }
    virtual void apply(int batch_size) {

        for(int i = 0; i < tunable_values.size(); i++){
            adam<DEVICE>(tunable_values[i]->values,
                         tunable_values[i]->gradients,
                         first_moments [i],
                         second_moments[i],
                         alpha, beta1, beta2, eps);
        }
    }
    virtual void newEpoch() {}
    virtual void logOverview() {}
};

#endif    // CUDAD_SRC_OPTIMIZER_ADAM_H_
