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