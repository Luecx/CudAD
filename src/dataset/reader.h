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

#ifndef BINARYPOSITIONWRAPPER_SRC_DATASET_READER_H_
#define BINARYPOSITIONWRAPPER_SRC_DATASET_READER_H_

#include "../position/fenparsing.h"
#include "../position/position.h"
#include "dataset.h"
#include "header.h"
#include "io.h"

#include <cmath>
#include <filesystem>
#include <string>
#include <sys/stat.h>

template<Format format>
inline DataSet read(const std::string& file, uint64_t count = -1) {

    constexpr uint64_t CHUNK_SIZE = (1 << 20);

    // open the file
    FILE* f;
    if (format == BINARY) {
        f = fopen(file.c_str(), "rb");
    } else if (format == TEXT) {
        f = fopen(file.c_str(), "r");
    }

    // create the dataset
    DataSet data_set {};

    // check if opening has worked
    if (f == nullptr) {
        std::cout << "could not open: " << file << std::endl;
        return DataSet {};
    }

    if (format == BINARY) {

        // read the header
        fread(&data_set.header, sizeof(Header), 1, f);

        // compute how much data to read
        auto data_to_read = std::min(count, data_set.header.position_count);
        data_set.positions.resize(data_to_read);
        int chunks = std::ceil(data_to_read / (float) CHUNK_SIZE);

        // actually load
        for (int c = 0; c < chunks; c++) {

            int start = c * CHUNK_SIZE;
            int end   = c * CHUNK_SIZE + CHUNK_SIZE;
            if (end > data_set.positions.size())
                end = data_set.positions.size();
            fread(&data_set.positions[start], sizeof(Position), end - start, f);
            printf("\r[Reading positions] Current count=%d", end);
            fflush(stdout);
        }
        std::cout << std::endl;
    } else if (format == TEXT) {

        int  c = 0;
        char buffer[128];
        while (fgets(buffer, 128, f) && (c++) < count) {
            // Remove trailing newline
            buffer[strcspn(buffer, "\n")] = 0;
            if (c % CHUNK_SIZE == 0) {
                printf("\r[Reading positions] Current count=%d", c);
                fflush(stdout);
            }
            data_set.positions.push_back(parseFen(std::string(buffer)));
        }
        printf("\r[Reading positions] Current count=%d", c - 1);
        fflush(stdout);

        std::cout << std::endl;
    }

    fclose(f);
    return data_set;
}

template<Format format>
inline bool isReadable(const std::string& file) {
    //    if (!std::filesystem::exists(file)) return false;

    if (format == BINARY) {
        std::filesystem::path p {file};
        auto                  size = std::filesystem::file_size(p);

        FILE*                 f;
        f = fopen(file.c_str(), "rb");
        if (f == nullptr) {
            return false;
        }

        Header header {};
        fread(&header, sizeof(Header), 1, f);

        auto expected_size = header.position_count * sizeof(Position) + sizeof(Header);

        return expected_size == size;

    } else {
        return true;
    }
}

#endif    // BINARYPOSITIONWRAPPER_SRC_DATASET_READER_H_
