
//
// Created by Luecx on 08.12.2021.
//

#ifndef BINARYPOSITIONWRAPPER_SRC_DATASET_BATCHLOADER_H_
#define BINARYPOSITIONWRAPPER_SRC_DATASET_BATCHLOADER_H_

#include "../misc/logging.h"
#include "../position/fenparsing.h"
#include "dataset.h"
#include "io.h"
#include "reader.h"

#include <fstream>
#include <random>
#include <utility>
#include <thread>

struct BatchLoader {
    int                      batch_size;
    int                      batches_per_load;
    volatile bool            next_batch_loaded;
    // The dataset being actively read from
    DataSet                  active_batch;
    // The next batch to be read from
    DataSet                  background_batch;

    DataSet*                 load_buffer;

    // track which part of the buffer we look at
    int                      current_batch_index = 0;

    // files to load
    std::vector<std::string> files {};
    std::ifstream            file {};
    int                      positions_left_in_file = 0;
    int                      current_file_index     = 0;

    BatchLoader(std::vector<std::string> p_files, int batch_size, int batches_per_load)
        : batch_size(batch_size), batches_per_load(batches_per_load) {
        load_buffer            = new DataSet {};
        load_buffer       ->positions.resize(static_cast<uint64_t>(capacity()));
        load_buffer       ->header.position_count = capacity();

        active_batch       .positions.resize(static_cast<uint64_t>(batch_size));
        active_batch       .header.position_count = batch_size;
        
        background_batch   .positions.resize(static_cast<uint64_t>(batch_size));
        background_batch   .header.position_count = batch_size;

        files                  = std::move(p_files);
        current_file_index     = -1;
        positions_left_in_file = 0;
        next_batch_loaded      = false;

        files.erase(
            std::remove_if(
                files.begin(),
                files.end(),
                [](const std::string& s){return !isReadable<BINARY>(s);}),
            files.end());

        if (files.size() == 0) {
            logging::write("Cannot create BatchLoader with no valid files. EXITING");
            exit(-1);
        }

        fillBuffer();
        std::thread t1(&BatchLoader::backgroundBatchLoading, this);
        t1.detach();
    }

    virtual ~BatchLoader() {
        delete load_buffer;
    }

    int capacity() const { return batch_size * batches_per_load; }

    void openNextFile() {
        if (file.is_open()) file.close();

        while (!file.is_open()) {
            current_file_index = (current_file_index + 1) % files.size();
            
            file = std::ifstream {files[current_file_index], std::ios::binary};

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
        int fens_to_fill = capacity();
        int read_offset       = 0;

        while (fens_to_fill > 0) {
            if (positions_left_in_file == 0) openNextFile();

            // read as many positions as possible from current file
            int filling             = std::min(fens_to_fill, positions_left_in_file);
            positions_left_in_file -= filling;

            file.read(reinterpret_cast<char*>(&load_buffer->positions[read_offset]),
                      sizeof(Position) * filling);

            if(file.gcount() != sizeof(Position) * filling) {
                logging::write("Some issue occured while reading file");
                exit(-1);
            }

            read_offset  += filling;
            fens_to_fill -= filling;
        }
    }

    void backgroundBatchLoading() {
        while (true) {
            // we've looped through our entire loading buffer
            if (current_batch_index >= batches_per_load) {
                fillBuffer();
                current_batch_index = 0;
            }

            // copy from load buffer into our background batch
            auto start = load_buffer->positions.begin() + current_batch_index * batch_size;
            auto end = start + batch_size;
            background_batch.positions.assign(start, end);
            
            current_batch_index++;
            
            // mark batch as loaded and wait
            next_batch_loaded = 1;
            while (next_batch_loaded);
        }
    }

    DataSet* next() {
        // wait until loaded
        while (!next_batch_loaded);

        active_batch.positions.assign(background_batch.positions.begin(), background_batch.positions.end());
        next_batch_loaded = false;
        return &active_batch;
    }
};

#endif    // BINARYPOSITIONWRAPPER_SRC_DATASET_BATCHLOADER_H_
