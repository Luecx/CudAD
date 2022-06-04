
//
// Created by Luecx on 26.11.2021.
//

#ifndef BINARYPOSITIONWRAPPER__PIECEBOARD_H_
#define BINARYPOSITIONWRAPPER__PIECEBOARD_H_

#include "bitboard.h"
#include "piece.h"
#include "piecelist.h"
#include "positionmeta.h"
#include "result.h"

#include <iomanip>
#include <ostream>

struct Position {

    PieceList               m_pieces {};
    BB                      m_occupancy {};
    PositionMetaInformation m_meta {};
    Result                  m_result {};

    template<Color color>
    Square getKingSquare() {
        if (color == WHITE) {
            return bitscanForwardIndex(m_occupancy, m_pieces.template bitscanPiece<WHITE_KING>());
        } else {
            return bitscanForwardIndex(m_occupancy, m_pieces.template bitscanPiece<BLACK_KING>());
        }
    }

    int    getPieceCount() const { return bitCount(m_occupancy); }

    Square getSquare(int piece_index) const { return bitscanForwardIndex(m_occupancy, piece_index); }

    Piece  getPiece(Square square) const {
        if (getBit(m_occupancy, square)) {
            return m_pieces.getPiece(bitCount(m_occupancy, square));
        }
        return NO_PIECE;
    }
};

#endif    // BINARYPOSITIONWRAPPER__PIECEBOARD_H_
