/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
