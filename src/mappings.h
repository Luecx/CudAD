//
// Created by Luecx on 03.03.2021.
//

#ifndef DIFFERENTIATION_MAPPINGS_H
#define DIFFERENTIATION_MAPPINGS_H

#include "dataset/dataset.h"
#include "position/fenparsing.h"
#include "data/SparseInput.h"
#include "data/SArray.h"

#include <cmath>

namespace dense_relative {



inline int king_square_index(Square relative_king_square){

    constexpr int indices[N_SQUARES]{
        0 ,1 ,2 ,3 ,3 ,2 ,1 ,0 ,
        4 ,5 ,6 ,7 ,7 ,6 ,5 ,4 ,
        8 ,9 ,10,11,11,10,9 ,8 ,
        8 ,9 ,10,11,11,10,9 ,8 ,
        12,12,13,13,13,13,12,12,
        12,12,13,13,13,13,12,12,
        14,14,15,15,15,15,14,14,
        14,14,15,15,15,15,14,14,
    };

    return indices[relative_king_square];


//    if (fileIndex(relative_king_square) > 3){
//        relative_king_square = mirrorHorizontally(relative_king_square);
//    }
//    return rankIndex(relative_king_square) * 4 + fileIndex(relative_king_square);
}

inline int index(Square psq, Piece p, Square kingSquare, Color view) {
//    constexpr int pieceTypeFactor  = 64;
//    constexpr int pieceColorFactor = 64 * 6;
//    constexpr int kingSideFactor   = 64 * 6 * 2;
//
//    const Square  relativeSquare    = view == WHITE ? psq : mirrorVertically(psq);
//    const PieceType pieceType       = p % 8;
//    const Color pieceColor          = p >= BLACK_PAWN ? BLACK : WHITE;
//    const bool kingSide             = (kingSquare & 7) > 3;
//
//    return relativeSquare
//           + pieceType * pieceTypeFactor
//           + (pieceColor == view) * pieceColorFactor
//           + kingSide * kingSideFactor;

    constexpr int pieceTypeFactor  = 64;
    constexpr int pieceColorFactor = 64 * 6;
    constexpr int kingSquareFactor = 64 * 6 * 2;

    const PieceType pieceType          = getPieceType(p);
    const Color     pieceColor         = getPieceColor(p);
    const Square    relativeKingSquare = view == WHITE ? kingSquare : mirrorVertically(kingSquare);
    const bool      kingSide           = fileIndex(kingSquare) > 3;
    const int       kingSquareIndex    = king_square_index(relativeKingSquare);
    Square          relativeSquare     = view == WHITE ? psq : mirrorVertically(psq);

    if (kingSide) {
        relativeSquare = mirrorHorizontally(relativeSquare);
    }

    return relativeSquare
           + pieceType              * pieceTypeFactor
           + (pieceColor == view)   * pieceColorFactor
           + kingSquareIndex        * kingSquareFactor;
}

inline void assign_input(Position& p, SparseInput& in1, SparseInput& in2, SArray<float>& output, int id) {


    float p_value  = p.m_result.score;
    float w_value  = p.m_result.wdl;

    // flip if black is to move -> relative network style
    if(p.m_meta.getActivePlayer() == BLACK){
        p_value = -p_value;
        w_value = -w_value;
    }

    float p_target = 1 / (1 + expf(-p_value * 2.5 / 400.0f));
    float w_target = (w_value + 1) / 2.0f;

    output(id) = (p_target + w_target) / 2;

    // track king squares
    Square wKingSq = p.getKingSquare<WHITE>();
    Square bKingSq = p.getKingSquare<BLACK>();

    BB bb{p.m_occupancy};
    int idx = 0;


    while(bb){
        Square sq  = bitscanForward(bb);
        Piece  pc = p.m_pieces.getPiece(idx);

        auto piece_index_white_pov = index(sq, pc, wKingSq, WHITE);
        auto piece_index_black_pov = index(sq, pc, bKingSq, BLACK);

        if (p.m_meta.getActivePlayer() == WHITE) {
            in1.set(id, piece_index_white_pov);
            in2.set(id, piece_index_black_pov);
        } else {
            in2.set(id, piece_index_white_pov);
            in1.set(id, piece_index_black_pov);
        }

        bb = lsbReset(bb);
        idx ++;
    }
}

inline void assign_inputs_batch(DataSet& positions, SparseInput& in1, SparseInput& in2,  SArray<float>& output) {

    ASSERT(positions.positions.size() == in1.n);
    ASSERT(positions.positions.size() == in2.n);
    ASSERT(positions.positions.size() == output.size);

    in1.clear();
    in2.clear();

#pragma omp parallel for schedule(static) num_threads(8)
    for (int i = 0; i < output.size; i++) {
        assign_input(positions.positions[i], in1,in2, output, i);
    }
}
}    // namespace dense_relative

#endif    // DIFFERENTIATION_MAPPINGS_H
