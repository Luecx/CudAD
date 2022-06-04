
//
// Created by Luecx on 27.11.2021.
//

#ifndef BINARYPOSITIONWRAPPER_SRC_RESULT_H_
#define BINARYPOSITIONWRAPPER_SRC_RESULT_H_

enum GameResult { WIN = 1, DRAW = 0, LOSS = -1 };

struct Result {
    int16_t score;
    int8_t  wdl;
};

#endif    // BINARYPOSITIONWRAPPER_SRC_RESULT_H_
