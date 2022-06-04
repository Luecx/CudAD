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

#ifndef BINARYPOSITIONWRAPPER_SRC_DATASET_HEADER_H_
#define BINARYPOSITIONWRAPPER_SRC_DATASET_HEADER_H_

#include <cstring>
#include <string>
struct Header {

    uint64_t position_count;

    char     engine_1[128];
    char     engine_2[128];
    char     comments[1024];

    // some setters
    void setEngine1(const std::string& string) {
        std::memset(engine_1, ' ', sizeof(char) * 128);
        for (int i = 0; i < std::min((int) string.size(), 128); i++) {
            engine_1[i] = string[i];
        }
        engine_1[127] = 0;
    }
    void setEngine2(const std::string& string) {
        std::memset(engine_2, ' ', sizeof(char) * 128);
        for (int i = 0; i < std::min((int) string.size(), 128); i++) {
            engine_2[i] = string[i];
        }
        engine_2[127] = 0;
    }
    void setComment(const std::string& string) {
        std::memset(comments, ' ', sizeof(char) * 1024);
        for (int i = 0; i < std::min((int) string.size(), 1024); i++) {
            comments[i] = string[i];
        }
        comments[1023] = 0;
    }

    uint64_t getPositionCount() const { return position_count; }
    void     setPositionCount(uint64_t position_count) { Header::position_count = position_count; }
};

#endif    // BINARYPOSITIONWRAPPER_SRC_DATASET_HEADER_H_
