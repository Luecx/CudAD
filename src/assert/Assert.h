

//
// Created by Luecx on 13.01.2022.
//

#ifndef CUDAD_SRC_ASSERT_ASSERT_H_
#define CUDAD_SRC_ASSERT_ASSERT_H_

#include <iostream>

#ifdef NDEBUG
#define ASSERT(expr)
#else
#define ASSERT(expr)                                                                                 \
    {                                                                                                \
        if (!static_cast<bool>(expr)) {                                                              \
            std::cout << "[ASSERT] in expression " << (#expr) << std::endl;                          \
            std::cout << "    file: " << __FILE__ << std::endl;                                      \
            std::cout << "    line: " << __LINE__ << std::endl;                                      \
            std::cout << "    func: " << __FUNCTION__ << std::endl;                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    }
#endif

#endif    // CUDAD_SRC_ASSERT_ASSERT_H_
