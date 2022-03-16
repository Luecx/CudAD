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

namespace dense_halo{


inline int index(Square psq, Piece p) {
    return psq + getPieceType(p) * 64 + !getPieceColor(p) * 64 * 6;
}

inline void assign_input(Position& p, SparseInput& in1, SArray<float>& output, SArray<bool>& output_mask, int id) {

    BB bb{p.m_occupancy};
    int idx = 0;

    while(bb){
        Square sq = bitscanForward(bb);
        Piece  pc = p.m_pieces.getPiece(idx);

        auto index_white = index(sq, pc);

        in1.set(id, index_white);

        bb = lsbReset(bb);
        idx ++;
    }

    float p_value  = p.m_result.score;
    float w_value  = p.m_result.wdl;

    float p_target = 1 / (1 + expf(-p_value * 2.5 / 400.0f));
    float w_target = (w_value + 1) / 2.0f;

    output      (id) = (p_target + w_target) / 2;
    output_mask (id) = true;
}

inline void assign_inputs_batch(DataSet& positions, SparseInput& in1, SArray<float>& output, SArray<bool>& output_mask) {

    ASSERT(positions.positions.size() == in1.n);
    in1.clear();
    output_mask.clear();

#pragma omp parallel for schedule(static) num_threads(16)
    for (int i = 0; i < positions.positions.size(); i++) {
        assign_input(positions.positions[i], in1, output, output_mask, i);
    }
}
}


namespace dense_relative_halo{

inline int index(Square psq, Piece p, Color view) {


    Square sq           = view == WHITE ? psq : mirrorVertically(psq);
    Color relativeColor = view == getPieceColor(p);
    PieceType pieceType = getPieceType(p);

    return sq
           + pieceType * 64
           + relativeColor * (64 * 6);

//    constexpr int pieceTypeFactor  = 64;
//    constexpr int pieceColorFactor = 64 * 6;
//
//    const PieceType pieceType          = getPieceType(p);
//    const Color     pieceColor         = getPieceColor(p);
//    Square          relativeSquare     = view == WHITE ? psq : mirrorVertically(psq);
//
//    return relativeSquare
//           + pieceType              * pieceTypeFactor
//           + (pieceColor != view)   * pieceColorFactor;
}

inline void assign_input(Position& p, SparseInput& in1, SparseInput& in2,SArray<float>& output, SArray<bool>& output_mask, int id) {

    BB bb{p.m_occupancy};
    int idx = 0;

    while(bb){
        Square sq = bitscanForward(bb);
        Piece  pc = p.m_pieces.getPiece(idx);

        auto index_white = index(sq, pc, WHITE);
        auto index_black = index(sq, pc, BLACK);

        if (p.m_meta.getActivePlayer() == WHITE) {
            in1.set(id, index_white);
            in2.set(id, index_black);
        } else {
            in2.set(id, index_white);
            in1.set(id, index_black);
        }

        bb = lsbReset(bb);
        idx ++;
    }

    float p_value  = p.m_result.score;
    float w_value  = p.m_result.wdl;

    // flip if black is to move -> relative network style
    if(p.m_meta.getActivePlayer() == BLACK){
        p_value = -p_value;
        w_value = -w_value;
    }

    float p_target = 1 / (1 + expf(-p_value * 2.5 / 400.0f));
    float w_target = (w_value + 1) / 2.0f;

    output      (id) = (p_target + w_target) / 2;
    output_mask (id) = true;
}

inline void assign_inputs_batch(DataSet& positions, SparseInput& in1,  SparseInput& in2, SArray<float>& output, SArray<bool>& output_mask) {

    ASSERT(positions.positions.size() == in1.n);
    in1.clear();
    in2.clear();
    output_mask.clear();

#pragma omp parallel for schedule(static) num_threads(16)
    for (int i = 0; i < positions.positions.size(); i++) {
        assign_input(positions.positions[i], in1, in2, output, output_mask, i);
    }
}
}

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

inline void assign_input(Position& p, SparseInput& in1, SparseInput& in2, SArray<float>& output, SArray<bool>& output_mask, int id) {

    constexpr static float phase_values[6]{0,1,1,2,4,0};


    // track king squares
    Square wKingSq = p.getKingSquare<WHITE>();
    Square bKingSq = p.getKingSquare<BLACK>();

    BB bb{p.m_occupancy};
    int idx = 0;
    int phase = 24;

    while(bb){
        Square sq = bitscanForward(bb);
        Piece  pc = p.m_pieces.getPiece(idx);

        auto piece_index_white_pov = index(sq, pc, wKingSq, WHITE);
        auto piece_index_black_pov = index(sq, pc, bKingSq, BLACK);

        phase -= phase_values[getPieceType(pc)];

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

    float p_value  = p.m_result.score;
    float w_value  = p.m_result.wdl;

    // flip if black is to move -> relative network style
    if(p.m_meta.getActivePlayer() == BLACK){
        p_value = -p_value;
        w_value = -w_value;
    }

    float p_target = 1 / (1 + expf(-p_value * 2.5 / 400.0f));
    float w_target = (w_value + 1) / 2.0f;

    int output_bucket = (bitCount(p.m_occupancy) - 1) / 4;

    output      (id) = (p_target + w_target) / 2;
    output_mask (id) = true;
}

inline void assign_inputs_batch(DataSet& positions, SparseInput& in1, SparseInput& in2,  SArray<float>& output, SArray<bool>& output_mask) {

    ASSERT(positions.positions.size() == in1.n);
    ASSERT(positions.positions.size() == in2.n);

    in1.clear();
    in2.clear();
    output_mask.clear();

#pragma omp parallel for schedule(static) num_threads(16)
    for (int i = 0; i < positions.positions.size(); i++) {
        assign_input(positions.positions[i], in1,in2, output, output_mask, i);
    }
}
}    // namespace dense_relative

#endif    // DIFFERENTIATION_MAPPINGS_H
