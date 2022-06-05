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

#ifndef DIFFERENTIATION_SRC_DATASET_SHUFFLE_H_
#define DIFFERENTIATION_SRC_DATASET_SHUFFLE_H_

#include "writer.h"

#include <cstring>
#include <vector>
#include <regex>

inline void shuffle(std::vector<std::string>& files) {

    int file_count = files.size();

    for (int i = 0; i < file_count; i++) {
        // create the dataset which will contain every (i + n % file_count) entry
        DataSet ds {};

        // loop over all datasets and append the corresponding element
        for (auto h : files) {
            DataSet org = read<BINARY>(h);

            // add all the elements evenly spaces with file_count
            for (int idx = i; idx < org.header.position_count; idx += file_count) {
                ds.positions.push_back(org.positions[idx]);
            }
        }

        ds.header.position_count = ds.positions.size();

        // shuffle
        ds.shuffle();

        write("shuffled_" + files[i], ds);
        std::cout << "wrote: "
                  << "shuffled_" + files[i] << std::endl;
    }
}

inline void mix_and_shuffle(std::vector<std::string>& files,
                            const std::string         out_dir,
                            const int                 num_files = 10) {
    for (int i = 0; i < files.size(); i++) {
        std::cout << "Reading from " << files[i] << std::endl;
        DataSet ds = read<BINARY>(files[i]);

        std::cout << "Shuffling " << ds.header.position_count << " positions" << std::endl;
        ds.shuffle();

        std::cout << "Writing to " << files[i] << std::endl;
        write(files[i], ds);
    }

    uint64_t              total_positions      = 0;
    uint64_t              files_with_positions = 0;
    std::vector<FILE*>    inputs {files.size()};
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

    std::cout << "There is a total of " << total_positions << " across " << files_with_positions
              << " files" << std::endl;

    std::srand(time(NULL));
    uint64_t number_out_files   = 10;
    uint64_t positions_per_file = total_positions / number_out_files;

    std::cout << "Creating " << number_out_files << " files with " << positions_per_file
              << " positions each" << std::endl;

    Position p[1];
    for (int i = 0; i < number_out_files; i++) {
        std::string out_name = out_dir + "berserk9dev2.d9." + std::to_string(i) + ".bin";

        std::cout << "Writing to " << out_name << std::endl;
        FILE*  fout = fopen(out_name.c_str(), "wb");

        Header header {};
        header.position_count = positions_per_file;
        fwrite(&header, sizeof(Header), 1, fout);

        for (uint64_t j = 1; j <= positions_per_file; j++) {
            if (j % 1000000 == 0) {
                std::printf("Wrote %10lld of %10lld to %s\r",
                            j,
                            positions_per_file,
                            out_name.c_str());
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

        std::printf("Wrote %10lld of %10lld to %s",
                    positions_per_file,
                    positions_per_file,
                    out_name.c_str());
        std::cout << std::endl;

        fclose(fout);
    }
}

/**
 * shuffles all the files and writes num_files output files.
 * The output files will be generated using the out_format.
 * It assumes the out_format contains at least one "$" (dollar sign) which will be replaced
 * with a number ranging from 1 to num_files
 * @param files
 * @param out_format
 * @param num_files
 */
inline void mix_and_shuffle_2(std::vector<std::string>& files,
                              const std::string&        out_format,
                              const int                 num_files = 32) {
    std::vector<FILE*> outfiles {};
    std::vector<int>   sizes {};

    outfiles.resize(num_files);
    sizes.resize(num_files);

    for (int i = 0; i < num_files; i++) {
        // replace the out_format's dollar signs with the index
        std::string file_name = out_format;
        file_name             = std::regex_replace(file_name, std::regex("\\$"), std::to_string(i + 1));

        // open the file and store it in the outfiles
        FILE* f     = fopen(file_name.c_str(), "wb");
        outfiles[i] = f;
        sizes[i]    = 0;

        // write the header
        Header header {};
        fwrite(&header, sizeof(Header), 1, f);
    }

    srand(time(NULL));

    // going through each file and writing the output files
    int count = 0;
    for (std::string s : files) {
        DataSet ds = read<BINARY>(s);
        for (Position& p : ds.positions) {
            // select a outfile
            int idx = rand() % num_files;
            // write it to the given file
            fwrite(&p, sizeof(Position), 1, outfiles[idx]);
            sizes[idx]++;
            count++;
        }
        // printing
        if (count % 1000000 == 0) {
            std::cout << count << std::endl;
        }

    }

    for (int i = 0; i < num_files; i++) {
        // correcting the size and closing the file
        // seek to the beginning
        fseek(outfiles[i], 0, SEEK_SET);
        // create new header and set position count
        Header header {};
        header.position_count = sizes[i];
        // overwrite the header at the start
        fwrite(&header, sizeof(Header), 1, outfiles[i]);
        // close
        fclose(outfiles[i]);
    }

    // final intra-file shuffling
    for (int i = 0; i < num_files; i++) {
        // regenerate the file name
        std::string file_name = out_format;
        file_name             = std::regex_replace(file_name, std::regex("\\$"), std::to_string(i + 1));
        // read
        DataSet ds = read<BINARY>(file_name);
        // shuffle
        ds.shuffle();
        // write
        write(file_name, ds);
    }
}

#endif    // DIFFERENTIATION_SRC_DATASET_SHUFFLE_H_
