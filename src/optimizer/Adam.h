//
// Created by Luecx on 16.01.2022.
//

#ifndef CUDAD_SRC_OPTIMIZER_ADAM_H_
#define CUDAD_SRC_OPTIMIZER_ADAM_H_

#include "../operations/adam/adam.h"
#include "../operations/clamp/clamp.h"
#include "Optimiser.h"

#include <tuple>

struct Adam : Optimiser {
    private:
    std::vector<SArray<float>>            first_moments {};
    std::vector<SArray<float>>            second_moments {};
    std::vector<std::tuple<float, float>> value_ranges {};

    public:
    double       alpha = 0.01;
    double       beta1 = 0.95;
    double       beta2 = 0.999;
    double       eps   = 1e-8;

    virtual void createBuffers() {
        for (Tape* t : tunable_values) {
            // clang-format off
            first_moments .push_back(SArray<float> {t->values.size});
            second_moments.push_back(SArray<float> {t->values.size});
            first_moments [first_moments.size() - 1].malloc_gpu();
            second_moments[first_moments.size() - 1].malloc_gpu();
            // clang-format on
            value_ranges.push_back(
                std::tuple<float, float> {t->min_allowed_value, t->max_allowed_value});
        }
    }
    virtual void apply(int batch_size) {

        for (int i = 0; i < tunable_values.size(); i++) {
            adam<DEVICE>(tunable_values[i]->values,
                         tunable_values[i]->gradients,
                         first_moments[i],
                         second_moments[i],
                         alpha,
                         beta1,
                         beta2,
                         eps);

            auto range = value_ranges[i];
            auto min   = std::get<0>(range);
            auto max   = std::get<1>(range);
            if (min != std::numeric_limits<float>::min()
                || max != std::numeric_limits<float>::max()) {
                clamp<DEVICE>(tunable_values[i]->values, min, max);
            }
        }
    }
    virtual void newEpoch() {}
    virtual void logOverview() {}
};

#endif    // CUDAD_SRC_OPTIMIZER_ADAM_H_
