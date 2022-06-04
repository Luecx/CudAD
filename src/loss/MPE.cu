
//
// Created by Luecx on 04.03.2022.
//

#include "../misc/logging.h"
#include "MPE.h"

// clang-format off
void MPE::apply(const SArray<float>& output,
                      SArray<float>& output_grad,
                const SArray<float>& target,
                const SArray<bool >& target_mask,
                Mode                 mode) {
    // clang-format on
    if (mode == HOST) {
        mpe<HOST>(output, output_grad, target, target_mask, loss, m_power, m_avg_gradients);
    } else {
        mpe<DEVICE>(output, output_grad, target, target_mask, loss, m_power, m_avg_gradients);
    }
}
void           MPE::logOverview() { logging::write("MPE(power=" + std::to_string(m_power) + ")"); }
// extract the loss
SArray<float>& MPE::getLoss() { return loss; }
// constructor
MPE::MPE(float power, bool avg_gradients) {
    m_power         = power;
    m_avg_gradients = avg_gradients;
    loss.malloc_gpu();
    loss.malloc_cpu();
}
