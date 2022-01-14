
//
// Created by Luecx on 28.10.2021.
//

#ifndef DIFFERENTIATION_SRC_LOGGING_H_
#define DIFFERENTIATION_SRC_LOGGING_H_

#include <fstream>
#include <string>

namespace logging {

// log file which we write to
extern std::ofstream log_file;

// some writen, open etc functions
void          write(const std::string& msg, const std::string& end="\n");
bool          isOpen();
void          open(const std::string& path);
void          close();
}    // namespace logging

#endif    // DIFFERENTIATION_SRC_LOGGING_H_
