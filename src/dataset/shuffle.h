

#ifndef DIFFERENTIATION_SRC_DATASET_SHUFFLE_H_
#define DIFFERENTIATION_SRC_DATASET_SHUFFLE_H_

#include "writer.h"

#include <cstring>
#include <vector>

inline void shuffle(std::vector<std::string>& files){

    int file_count = files.size();

    for(int i = 0; i < file_count; i++){
        // create the dataset which will contain every (i + n % file_count) entry
        DataSet ds{};


        // loop over all datasets and append the corresponding element
        for(auto h:files){
            DataSet org = read<BINARY>(h);

            // add all the elements evenly spaces with file_count
            for(int idx = i; idx < org.header.position_count; idx+=file_count){
                ds.positions.push_back(org.positions[idx]);
            }
        }

        ds.header.position_count = ds.positions.size();

        // shuffle
        ds.shuffle();

        write("shuffled_" + files[i], ds);
        std::cout << "wrote: " << "shuffled_" + files[i] << std::endl;
    }
}

#endif    // DIFFERENTIATION_SRC_DATASET_SHUFFLE_H_
