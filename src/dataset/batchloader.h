
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

struct BatchLoader {

    int                      batch_size;
    int                      batches;
    DataSet                  batch;
    DataSet*                 data;
    DataSet*                 data_buffer;

    // track which part of the buffer we look at
    int                      current_batch_index = 0;

    // files to load
    std::vector<std::string> files {};
    std::ifstream            file {};
    int                      positions_left_in_file = 0;
    int                      current_file_index     = 0;

    BatchLoader(std::vector<std::string> p_files, int batch_size, int batches)
        : batch_size(batch_size), batches(batches) {
        data                   = new DataSet {};
        data_buffer            = new DataSet {};
        batch       .positions.resize(static_cast<uint64_t>(batch_size));
        data       ->positions.resize(static_cast<uint64_t>(capacity()));
        data_buffer->positions.resize(static_cast<uint64_t>(capacity()));
        batch       .header.position_count = batch_size;
        data       ->header.position_count = capacity();
        data_buffer->header.position_count = capacity();

        files                  = std::move(p_files);
        current_file_index     = -1;
        positions_left_in_file = 0;

        files.erase(
            std::remove_if(
                files.begin(),
                files.end(),
                [](const std::string& s){return !isReadable<BINARY>(s);}),
            files.end());

        if(files.size() == 0){
            logging::write("Cannot create BatchLoader with no valid files. EXITING");
            exit(-1);
        }

        fillBuffer();
        std::swap(data_buffer, data);
        fillBuffer();
    }

    virtual ~BatchLoader() {
        delete data;
        delete data_buffer;
    }

    int  capacity() const { return batch_size * batches; }

    void openNextFile() {

        if (file.is_open()) {
            file.close();
        }
        while (!file.is_open()) {
            current_file_index += 1;
            current_file_index %= files.size();
            file = std::ifstream {files[current_file_index], std::ios::binary};
            if (!file.is_open()) {
                logging::write("Could not open file: " + files[current_file_index]);
                file.close();
            }
        }

        Header header {};
        // read header
        file.read(reinterpret_cast<char*>(&header), sizeof(Header));

        // store how many fens to read from this file
        positions_left_in_file = header.position_count;

//        logging::write("Opened: "
//                       + files[current_file_index]
//                       + " and found "
//                       + std::to_string(positions_left_in_file)
//                       + " positions");
    }

    void fillBuffer() {

        // if no positions left in current file, open next one (most likely for the very first batch)
        if(positions_left_in_file == 0){
            openNextFile();
        }

        // store how many fens we still need to fill in
        int fens_to_fill = capacity();
        // compute the offset at which we are going to store the data
        int offset       = 0;
        // open files as long as we still need to fill fens
        while (fens_to_fill > 0) {

//            std::cout << "left in file: " << positions_left_in_file << std::endl;

            int filling             = std::min(fens_to_fill, positions_left_in_file);
            positions_left_in_file -= filling;
            file.read(reinterpret_cast<char*>(&data_buffer->positions[offset]),
                      sizeof(Position) * filling);

//            std::shuffle(data_buffer->positions.begin(), data_buffer->positions.end(), std::mt19937(std::random_device()()));

//            std::cout << "read  " << file.gcount() << " bytes" << std::endl;
//            std::cout << "tried " << sizeof(Position) * filling << " bytes" << std::endl;
//            std::cout << positions_left_in_file << std::endl;

            if(file.gcount() != sizeof(Position) * filling){
                logging::write("Some issue occured while reading file");
                exit(-1);
            }

            offset       += filling;
            fens_to_fill -= filling;

            // open the next file if we will loop
            if(fens_to_fill > 0)
                openNextFile();
        }
        if(positions_left_in_file == 0){
            openNextFile();
        }
    }

    DataSet* next(){

        // swap data buffer with data
        if(current_batch_index >= batches){
            std::swap(data_buffer, data);
            // fill the new buffer (potentially multithreading next?)
            fillBuffer();
            current_batch_index = 0;
        }

        // assign the data to the batch
        batch.positions.assign(data->positions.begin() + current_batch_index * batch_size, data->positions.begin() + current_batch_index * batch_size + batch_size);
        // copy the data to the batch
//        std::memcpy(&batch.positions[0],
//                    &data->positions[current_batch_index * batch_size],
//                    sizeof(Position) * batch_size);

        current_batch_index ++;
        return &batch;
    }
};

#endif    // BINARYPOSITIONWRAPPER_SRC_DATASET_BATCHLOADER_H_
