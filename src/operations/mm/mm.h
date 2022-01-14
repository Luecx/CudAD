

//
// Created by Luecx on 14.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MM_MM_H_
#define CUDAD_SRC_OPERATIONS_MM_MM_H_

#include "mm_cublas.h"

template<Mode mode>
inline void mm(DenseMatrix &mat1,
        DenseMatrix &mat2,
        DenseMatrix &res){
    ASSERT(mat1.n == mat2.m);
    ASSERT(mat1.m == res .m);
    ASSERT(mat2.n == res .n);

    if(mode == DEVICE){
        mm_cublas(mat1, mat2, res);
    }else{
        ASSERT(false);
    }
}

#endif //CUDAD_SRC_OPERATIONS_MM_MM_H_
