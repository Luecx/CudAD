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

inline int index(Square psq, Piece p, Square kingSquare, Color view) {

    Square    relativeSquare = view == WHITE ? psq : mirrorVertically(psq);
    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    PieceType pieceType      = p % 8;
    bool      kingSide       = (kingSquare & 7) > 3;

    if (kingSide) {
        relativeSquare ^= 7;
    }

    return relativeSquare
           + (pieceColor == view) * 64 * 6
           + pieceType * 64;
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
