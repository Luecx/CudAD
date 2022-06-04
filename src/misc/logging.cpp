
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
