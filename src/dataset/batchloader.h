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

#ifndef BINARYPOSITIONWRAPPER_SRC_DATASET_BATCHLOADER_H_
#define BINARYPOSITIONWRAPPER_SRC_DATASET_BATCHLOADER_H_

#include "../misc/logging.h"
#include "../position/fenparsing.h"
#include "dataset.h"
#include "io.h"
#include "reader.h"

#include <fstream>
#include <random>
#include <thread>
#include <utility>

struct BatchLoader {
    int           batch_size;
    volatile bool next_batch_loaded = false;
    // The dataset being actively read from
    DataSet  active_batch;

    DataSet* load_buffer;

    // files to load
    std::vector<std::string> files {};
    std::ifstream            file {};
    int                      positions_left_in_file = 0;
    int                      current_file_index     = 0;

    BatchLoader(std::vector<std::string> p_files, int batch_size, int validate_files = true)
        : batch_size(batch_size) {
        load_buffer = new DataSet {};
        load_buffer->positions.resize(static_cast<uint64_t>(batch_size));
        load_buffer->header.position_count = batch_size;

        active_batch.positions.resize(static_cast<uint64_t>(batch_size));
        active_batch.header.position_count = batch_size;

        files                              = std::move(p_files);
        current_file_index                 = -1;
        positions_left_in_file             = 0;
        next_batch_loaded                  = false;

        if (validate_files) {
            files.erase(std::remove_if(files.begin(),
                                       files.end(),
                                       [](const std::string& s) { return !isReadable<BINARY>(s); }),
                        files.end());
        }

        if (files.size() == 0) {
            logging::write("Cannot create BatchLoader with no valid files. EXITING");
            exit(-1);
        }

        std::thread t1(&BatchLoader::backgroundBatchLoading, this);
        t1.detach();
    }

    virtual ~BatchLoader() { delete load_buffer; }

    void openNextFile() {
        if (file.is_open())
            file.close();

        while (!file.is_open()) {
            current_file_index = (current_file_index + 1) % files.size();

            file               = std::ifstream {files[current_file_index], std::ios::binary};

            if (!file.is_open()) {
                logging::write("Could not open file: " + files[current_file_index]);
                file.close();
            }
        }

        // get the positions in file
        Header header {};
        file.read(reinterpret_cast<char*>(&header), sizeof(Header));
        positions_left_in_file = header.position_count;
    }

    void fillBuffer() {
        int fens_to_fill = batch_size;
        int read_offset  = 0;

        while (fens_to_fill > 0) {
            if (positions_left_in_file == 0)
                openNextFile();

            // read as many positions as possible from current file
            int filling = std::min(fens_to_fill, positions_left_in_file);
            positions_left_in_file -= filling;

            file.read(reinterpret_cast<char*>(&load_buffer->positions[read_offset]),
                      sizeof(Position) * filling);

            if (file.gcount() != sizeof(Position) * filling) {
                logging::write("Some issue occured while reading file");
                exit(-1);
            }

            read_offset += filling;
            fens_to_fill -= filling;
        }
    }

    void backgroundBatchLoading() {
        while (true) {
            fillBuffer();

            // mark batch as loaded and wait
            next_batch_loaded = true;
            while (next_batch_loaded)
                ;
        }
    }

    DataSet* next() {
        // wait until loaded
        while (!next_batch_loaded)
            ;

        active_batch.positions.assign(load_buffer->positions.begin(), load_buffer->positions.end());

        next_batch_loaded = false;
        return &active_batch;
    }
};

#endif    // BINARYPOSITIONWRAPPER_SRC_DATASET_BATCHLOADER_H_
