
//
// Created by Luecx on 27.11.2021.
//

#ifndef BINARYPOSITIONWRAPPER_SRC_POSITIONMETA_H_
#define BINARYPOSITIONWRAPPER_SRC_POSITIONMETA_H_

#include "bitboard.h"
#include "square.h"

struct PositionMetaInformation {

    uint8_t m_move_count;
    uint8_t m_fifty_move_rule;
    uint8_t m_castling_and_active_player {};
    Square  m_en_passant_square {N_SQUARES};

    Color   getActivePlayer() const { return getBit(m_castling_and_active_player, 7); }

    void    setActivePlayer(Color color) {
        if (color) {
            m_castling_and_active_player |= (1 << 7);
        } else {
            m_castling_and_active_player &= ~(1 << 7);
        }
    }

    Square getEnPassantSquare() const { return m_en_passant_square; }

    void   setEnPassantSquare(Square ep_square) { m_en_passant_square = ep_square; }

    bool   getCastlingRight(Color player, Side side) const {
        return m_castling_and_active_player & (1 << (player * 2 + side));
    }

    void setCastlingRight(Color player, Side side, bool value) {
        if (value)
            m_castling_and_active_player |= (1 << (player * 2 + side));
        else
            m_castling_and_active_player &= ~(1 << (player * 2 + side));
    }

    uint8_t getFiftyMoveRule() const { return m_fifty_move_rule; }
    void    setFiftyMoveRule(uint8_t p_fifty_move_rule) { m_fifty_move_rule = p_fifty_move_rule; }

    uint8_t getMoveCount() const { return m_move_count; }
    void    setMoveCount(uint8_t p_move_count) { m_move_count = p_move_count; }
};

#endif    // BINARYPOSITIONWRAPPER_SRC_POSITIONMETA_H_
