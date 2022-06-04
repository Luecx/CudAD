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

#include "logging.h"

#include "chrono"

#include <ctime>
#include <iostream>

std::ofstream logging::log_file {};

using namespace std::chrono;

void logging::write(const std::string& msg, const std::string& end) {
    if (!isOpen()) {
        std::cout << msg << end << std::flush;
    } else {
        log_file << msg << end << std::flush;
    }
}
bool logging::isOpen() { return log_file.is_open(); }
void logging::open(const std::string& path) {
    if (logging::isOpen()) {
        logging::close();
    }
    log_file = std::ofstream {path};
}
void logging::close() {
    if (logging::isOpen())
        log_file.close();
}
