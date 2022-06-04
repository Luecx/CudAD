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

#include <cmath>
#include <iostream>
// clang-format off
void clamp_host(
    float* values,
    float p_min,
    float p_max,
    float size){

    // clang-format on
    for (int idx = 0; idx < size; idx++) {

        if (idx >= size)
            return;

        values[idx] = std::max(p_min, std::min(p_max, values[idx]));
    }
}
