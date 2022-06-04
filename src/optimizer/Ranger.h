#ifndef CUDAD_SRC_OPTIMIZER_RANGER_H_
#define CUDAD_SRC_OPTIMIZER_RANGER_H_

#include <tuple>
#include "Optimiser.h"
#include "../operations/clamp/clamp.h"
#include "../operations/ranger/ranger.h"

struct Ranger : Optimiser {
    private:
        int step = 0;
        std::vector<SArray<float>>            exp_avg      {};
        std::vector<SArray<float>>            exp_avg_sq   {};
        std::vector<SArray<float>>            slow_buffer  {};
        std::vector<std::tuple<float, float>> value_ranges {};

    public:
        double lr              = 1e-3;
        double beta1           = 0.95;
        double beta2           = 0.999;
        double eps             = 1e-5;
        double alpha           = 0.5;
        int    k               = 6;
        int    N_sma_threshold = 6;
    
    virtual void createBuffers() {
        for (Tape* t: tunable_values) {
            exp_avg    .push_back(SArray<float>{ t->values.size });
            exp_avg_sq .push_back(SArray<float>{ t->values.size });
            slow_buffer.push_back(SArray<float>{ t->values.size });

            exp_avg    [exp_avg.size()     - 1].malloc_gpu();
            exp_avg_sq [exp_avg_sq.size()  - 1].malloc_gpu();
            slow_buffer[slow_buffer.size() - 1].malloc_gpu();

            value_ranges.push_back(std::tuple<float, float>{ t->min_allowed_value, t->max_allowed_value });
        }
    }

    virtual void apply(int batch_size) {
        step++;

        for (int i = 0; i < tunable_values.size(); i++) {
            ranger<DEVICE>(tunable_values[i]->values,
                           tunable_values[i]->gradients,
                           exp_avg[i],
                           exp_avg_sq[i],
                           slow_buffer[i],
                           step, lr, beta1, beta2, eps, 
                           alpha, k, N_sma_threshold);
            
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

#endif // CUDAD_SRC_OPTIMIZER_RANGER_H_