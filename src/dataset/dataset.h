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

#ifndef BINARYPOSITIONWRAPPER_SRC_DATASET_DATASET_H_
#define BINARYPOSITIONWRAPPER_SRC_DATASET_DATASET_H_

#include "../position/position.h"
#include "header.h"

#include <algorithm>
#include <random>
#include <vector>

struct DataSet {

    Header                header {};
    std::vector<Position> positions {};

    void                  addData(DataSet& other) {
        positions.insert(std::end(positions), std::begin(other.positions), std::end(other.positions));
        header.position_count += other.header.position_count;
    }

    void shuffle() {
        std::shuffle(positions.begin(), positions.end(), std::mt19937(std::random_device()()));
    }
};

#endif    // BINARYPOSITIONWRAPPER_SRC_DATASET_DATASET_H_
