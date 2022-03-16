
//
// Created by Luecx on 04.03.2022.
//

#include "MPE.h"
#include "../misc/logging.h"

void MPE::apply(const SArray<float>& output,
                      SArray<float>& output_grad,
                const SArray<float>& target,
                const SArray<bool >& target_mask,
                Mode                 mode) {
    if(mode == HOST){
        mpe<HOST>(output, output_grad, target, target_mask, loss, m_power);
    }else{
        mpe<DEVICE>(output, output_grad, target, target_mask, loss, m_power);
    }
}
void MPE::logOverview() {logging::write("MPE(power=" + std::to_string(m_power) + ")");}
SArray<float>& MPE::getLoss() { return loss; }
MPE::MPE(float power) {
    m_power = power;
    loss.malloc_gpu();
    loss.malloc_cpu();
}
