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

#ifndef BINARYPOSITIONWRAPPER_SRC_DEFS_H_
#define BINARYPOSITIONWRAPPER_SRC_DEFS_H_

#include <stdint.h>

typedef uint64_t BB;
typedef int8_t   Square;
typedef int8_t   Diagonal;
typedef int8_t   AntiDiagonal;
typedef int8_t   Direction;

typedef int8_t   File;
typedef int8_t   Rank;
typedef int8_t   Piece;
typedef int8_t   PieceType;
typedef bool     Side;
typedef bool     Color;
#endif    // BINARYPOSITIONWRAPPER_SRC_DEFS_H_
