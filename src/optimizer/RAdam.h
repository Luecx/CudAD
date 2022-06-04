#ifndef CUDAD_SRC_OPTIMIZER_RADAM_H_
#define CUDAD_SRC_OPTIMIZER_RADAM_H_

#include <tuple>
#include "Optimiser.h"
#include "../operations/clamp/clamp.h"
#include "../operations/radam/radam.h"

struct RAdam : Optimiser {
    private:
        int step = 0;
        std::vector<SArray<float>>            exp_avg      {};
        std::vector<SArray<float>>            exp_avg_sq   {};
        std::vector<std::tuple<float, float>> value_ranges {};

    public:
        double lr              = 1e-3;
        double beta1           = 0.9;
        double beta2           = 0.999;
        double eps             = 1e-8;
        int    N_sma_threshold = 5;
    
    virtual void createBuffers() {
        for (Tape* t: tunable_values) {
            exp_avg   .push_back(SArray<float>{ t->values.size });
            exp_avg_sq.push_back(SArray<float>{ t->values.size });

            exp_avg   [exp_avg.size()    - 1].malloc_gpu();
            exp_avg_sq[exp_avg_sq.size() - 1].malloc_gpu();

            value_ranges.push_back(std::tuple<float, float>{ t->min_allowed_value, t->max_allowed_value });
        }
    }

    virtual void apply(int batch_size) {
        step++;

        for (int i = 0; i < tunable_values.size(); i++) {
            radam<DEVICE>(tunable_values[i]->values,
                          tunable_values[i]->gradients,
                          exp_avg[i],
                          exp_avg_sq[i],
                          step, lr, beta1, beta2, eps, N_sma_threshold);
            
            auto range = value_ranges[i];
            auto min = std::get<0>(range);
            auto max = std::get<1>(range);
            
            if (min != std::numeric_limits<float>::min() ||
                max != std::numeric_limits<float>::max())
                clamp<DEVICE>(tunable_values[i]->values, min, max);
        }
    }

    virtual void newEpoch() {}
    virtual void logOverview() {}
};

#endif // CUDAD_SRC_OPTIMIZER_RADAM_H_