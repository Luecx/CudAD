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

#ifndef DIFFERENTIATION_SRC_LOGGING_H_
#define DIFFERENTIATION_SRC_LOGGING_H_

#include <fstream>
#include <string>

namespace logging {

// log file which we write to
extern std::ofstream log_file;

// some writen, open etc functions
void write(const std::string& msg, const std::string& end = "\n");
bool isOpen();
void open(const std::string& path);
void close();
}    // namespace logging

#endif    // DIFFERENTIATION_SRC_LOGGING_H_
