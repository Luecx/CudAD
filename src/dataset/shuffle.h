

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

inline void mix_and_shuffle(std::vector<std::string>& files, const std::string out_dir, const int num_files = 10) {
    for (int i = 0; i < files.size(); i++) {
        std::cout << "Reading from " << files[i] << std::endl;
        DataSet ds = read<BINARY>(files[i]);
        
        std::cout << "Shuffling " << ds.header.position_count << " positions" << std::endl;
        ds.shuffle();

        std::cout << "Writing to " << files[i] << std::endl;
        write(files[i], ds);
    }

    uint64_t total_positions = 0;
    uint64_t files_with_positions = 0;
    std::vector<FILE*> inputs {files.size()};
    std::vector<uint64_t> counts {files.size()};

    for (int i = 0; i < files.size(); i++) {
        std::cout << "Reading from " << files[i] << std::endl;

        FILE* fin = fopen(files[i].c_str(), "rb");
        inputs[i] = fin;

        Header header {};
        fread(&header, sizeof(Header), 1, fin);

        total_positions += header.position_count;
        counts[i] = header.position_count;
        files_with_positions++;
    }

    std::cout << "There is a total of " << total_positions << " across " << files_with_positions << " files" << std::endl;

    std::srand(time(NULL));
    uint64_t number_out_files = 10;
    uint64_t positions_per_file = total_positions / number_out_files;

    std::cout << "Creating " << number_out_files << " files with " << positions_per_file << " positions each" << std::endl;

    Position p[1];
    for (int i = 0; i < number_out_files; i++) {
        std::string out_name = out_dir + "berserk9dev2.d9." + std::to_string(i) + ".bin";

        std::cout << "Writing to " << out_name << std::endl;
        FILE* fout = fopen(out_name.c_str(), "wb");
        
        Header header {};
        header.position_count = positions_per_file;
        fwrite(&header, sizeof(Header), 1, fout);

        for (uint64_t j = 1; j <= positions_per_file; j++) {
            if (j % 1000000 == 0) {
                std::printf("Wrote %10lld of %10lld to %s\r", j, positions_per_file, out_name.c_str());
                std::cout << std::flush;
            }

            int file_idx = files_with_positions > 1 ? std::rand() % files_with_positions : 0;
            fread(p, sizeof(Position), 1, inputs[file_idx]);
            fwrite(p, sizeof(Position), 1, fout);

            counts[file_idx]--;

            if (counts[file_idx] <= 0) {
                fclose(inputs[file_idx]);

                files_with_positions--;

                if (file_idx != files_with_positions) {
                    inputs[file_idx] = inputs[files_with_positions];
                    counts[file_idx] = counts[files_with_positions];
                }
            }
        }

        std::printf("Wrote %10lld of %10lld to %s", positions_per_file, positions_per_file, out_name.c_str());
        std::cout << std::endl;

        fclose(fout);
    }
}

#endif    // DIFFERENTIATION_SRC_DATASET_SHUFFLE_H_
