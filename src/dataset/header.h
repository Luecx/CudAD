
//
// Created by Luecx on 27.11.2021.
//

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
