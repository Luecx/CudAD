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

#ifndef BINARYPOSITIONWRAPPER_SRC_SQUARE_H_
#define BINARYPOSITIONWRAPPER_SRC_SQUARE_H_

// clang-format off
enum Squares {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    N_SQUARES = 64
};

constexpr char const* square_identifier[] {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
};

// clang-format on
inline Square squareIndex(Rank rank, File file) { return 8 * rank + file; }

inline Square squareIndex(std::string& str) {

    Rank r = str.at(1) - '1';
    File f = toupper(str.at(0)) - 'A';

    return squareIndex(r, f);
}

inline Rank   rankIndex(Square square_index) { return square_index >> 3; }

inline File   fileIndex(Square square_index) { return square_index & 7; }

inline Square mirrorVertically(Square square) { return square ^ 56; }

inline Square mirrorHorizontally(Square square) { return square ^ 7; }

inline Color  getSquareColor(Square square) {
    constexpr BB white_squares {0x55AA55AA55AA55AAULL};
    if (getBit(white_squares, square))
        return WHITE;
    return BLACK;
}

#endif    // BINARYPOSITIONWRAPPER_SRC_SQUARE_H_
