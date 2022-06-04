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

#ifndef DIFFERENTIATION_SRC_MISC_CSV_H_
#define DIFFERENTIATION_SRC_MISC_CSV_H_

#include <cstdarg>
#include <fstream>
struct CSVWriter {

    std::ofstream csv_file {};

    CSVWriter(std::string res) { csv_file = std::ofstream {res}; }

    virtual ~CSVWriter() { csv_file.close(); }

    void write(std::initializer_list<std::string> args) {
        for (auto col = args.begin(); col != args.end(); ++col) {
            if (col != args.begin())
                csv_file << ",";

            csv_file << "\"" << *col << "\"";
        }

        // new line and flush output
        csv_file << std::endl;
    }
};

#endif    // DIFFERENTIATION_SRC_MISC_CSV_H_