#include <iostream>
#include "data/SArray.h"
#include "data/DenseMatrix.h"
#include "operations/operations.h"
#include "misc/timer.h"


int main() {

    init();

    uint32_t B = 16384;
    uint32_t I = 768;
    uint32_t O = 512;

    DenseMatrix wgt    {O,I};
    DenseMatrix wgt_grd{O,I};
    DenseMatrix wgt_fm {O,I};
    DenseMatrix wgt_sm {O,I};

    SparseInput inp{I,B,32};

    DenseMatrix bia    {O,1};
    DenseMatrix bia_grd{O,1};

    DenseMatrix res    {O,B};
    DenseMatrix res_grd{O,B};

    DenseMatrix tar{O,B};

    for(int b = 0; b < B; b++){
        for(int k = 0; k < 32; k++){

            inp.set(b, rand() % I);
        }
    }

    wgt.randomise(-1,1);
    tar.randomise(-1,1);

    wgt.gpu_upload();
    bia.gpu_upload();
    res.gpu_upload();
    tar.gpu_upload();
    inp.column_indices.gpu_upload();

    wgt_grd.gpu_upload();
    bia_grd.gpu_upload();
    res_grd.gpu_upload();

    Timer t{};
    for(int i = 0; i < 1000; i++){
        sparse_affine<DEVICE>(wgt, inp, bia, res);

//        res.gpu_download();
//        float loss = 0;
//        for(int i = 0; i < res_grd.size; i++){
//            float diff = res(i) - tar(i);
//            loss += diff * diff;
//            res_grd(i) = 2 * diff;
//        }
//        std::cout << "loss= " << loss << std::endl;
//        res_grd.gpu_upload();
//
//        wgt_grd.clear<DEVICE>();
//        bia_grd.clear<DEVICE>();
        sparse_affine_bp<DEVICE>(wgt_grd, inp, bia_grd, res_grd);

//        wgt_grd.gpu_download();
//        wgt.gpu_download();
////        std::cout << wgt << std::endl;
////        std::cout << wgt_grd << std::endl;
//        add<DEVICE>(wgt, wgt_grd, wgt, 1, 0.01);
        adam<DEVICE>(wgt, wgt_grd, wgt_fm, wgt_sm, 0.001, 0.9, 0.999, 1e-8);

//        wgt.gpu_download();
////        std::cout << wgt << std::endl;
////        std::cout << wgt_grd << std::endl;
    }
    t.tock();
    std::cout << t.duration() << std::endl;

    inp.column_indices.gpu_download();
    wgt.gpu_download();
    bia.gpu_download();
    res.gpu_download();

//    std::cout << wgt << std::endl;
//    std::cout << bia << std::endl;
//    std::cout << res << std::endl;

    close();

//    dense_matrix(3,3) = 1;
//    dense_matrix(3) = 4;
//
//    DenseMatrix test{4,1};
//    for(int i = 0; i < 4; i++){
//        test(i) = i + 1;
//    }
//    std::cout << dense_matrix << " \n " << test << std::endl;
//    test.gpu_upload();
//
//    add<DEVICE>(dense_matrix, test, dense_matrix, 0,0.45);
//    dense_matrix.gpu_download();
//
//    std::cout << dense_matrix << std::endl;
//    for(int i = 0; i < 12; i++){
//        dense_matrix(i,i) = i;
//    }

//    DenseMatrix sub_matrix{6,6};
//    sub_matrix(0,0) = 1;
//
//
//    std::cout << dense_matrix.cpu_values << std::endl;
//    std::cout << sub_matrix.cpu_values << std::endl;
//    std::cout << dense_matrix << std::endl;
//    std::cout << sub_matrix << std::endl;

}
