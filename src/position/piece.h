
//
// Created by Luecx on 26.11.2021.
//

#ifndef BINARYPOSITIONWRAPPER__PIECE_H_
#define BINARYPOSITIONWRAPPER__PIECE_H_

#include "defs.h"

enum Colors { WHITE = false, BLACK = true, N_COLORS = 2 };

enum Sides { KING_SIDE, QUEEN_SIDE, N_SIDES = 2 };

enum PieceTypes {
    PAWN          = 0,
    KNIGHT        = 1,
    BISHOP        = 2,
    ROOK          = 3,
    QUEEN         = 4,
    KING          = 5,
    N_PIECE_TYPES = 6
};

enum Pieces {
    NO_PIECE     = -1,
    WHITE_PAWN   = 0,
    WHITE_KNIGHT = 1,
    WHITE_BISHOP = 2,
    WHITE_ROOK   = 3,
    WHITE_QUEEN  = 4,
    WHITE_KING   = 5,
    BLACK_PAWN   = 8,
    BLACK_KNIGHT = 9,
    BLACK_BISHOP = 10,
    BLACK_ROOK   = 11,
    BLACK_QUEEN  = 12,
    BLACK_KING   = 13,
    N_PIECES     = 14
};

constexpr char
    piece_identifier[] {'P', 'N', 'B', 'R', 'Q', 'K', ' ', ' ', 'p', 'n', 'b', 'r', 'q', 'k'};

inline Color     getPieceColor(Piece p) { return p & 0x8; }

inline PieceType getPieceType(Piece p) { return p & 0x7; }

inline Piece     getPiece(Color c, PieceType pt) { return c * 8 + pt; }

#endif    // BINARYPOSITIONWRAPPER__PIECE_H_
