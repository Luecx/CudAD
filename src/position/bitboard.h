
//
// Created by Luecx on 26.11.2021.
//

#ifndef BINARYPOSITIONWRAPPER__BITBOARD_H_
#define BINARYPOSITIONWRAPPER__BITBOARD_H_

#include "defs.h"

#include <iostream>
#ifndef __ARM__
#include <immintrin.h>
#endif

/**
 * toggles the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void   toggleBit(BB& number, Square index) { number ^= (1ULL << index); }

/**
 * set the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void   setBit(BB& number, Square index) { number |= (1ULL << index); }

/**
 * unset the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void   unsetBit(BB& number, Square index) { number &= ~(1ULL << index); }

/**
 * get the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline bool   getBit(BB number, Square index) { return ((number >> index) & 1ULL) == 1; }

/**
 * returns the index of the LSB
 * @param bb
 * @return
 */
inline Square bitscanForward(BB bb) {
    //    UCI_ASSERT(bb != 0);
    return _tzcnt_u64(bb);
}

/**
 * returns the index of the MSB
 * @param bb
 * @return
 */
inline Square bitscanReverse(BB bb) {
    //    UCI_ASSERT(bb != 0);
    ASSERT(false);
    //    return __builtin_clzll(bb) ^ 63;
}

/**
 * returns the index of the nth set bit, starting at the lsb
 * @param bb
 * @return
 */
inline Square bitscanForwardIndex(BB bb, Square n) {

#ifdef __ARM__
https:    // stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int
    n += 1;
    BB shifted = 0;    // running total
    BB nBits;          // value for this iteration

    // handle no solution
    if (n > bitCount(bb))
        return 64;

    while (n > 7) {
        // for large n shift out lower n-1 bits from v.
        nBits = n - 1;
        n -= bitCount(bb & ((1 << nBits) - 1));
        bb >>= nBits;
        shifted += nBits;
    }

    BB next;
    // n is now small, clear out n-1 bits and return the next bit
    // v&(v-1): a well known software trick to remove the lowest set bit.
    while (next = bb & (bb - 1), --n) {
        bb = next;
    }
    return bitscanForward((bb ^ next) << shifted);
#else
    return bitscanForward(_pdep_u64(1ULL << n, bb));
#endif
}

/**
 * returns the amount of set bits in the given bitboard.
 * @param bb
 * @return
 */
inline int bitCount(BB bb) { return __popcnt64(bb); }

/**
 * counts the ones inside the bitboard before the given index
 */
inline int bitCount(BB bb, int pos) {
    BB mask = ((BB) 1 << pos) - 1;
    return bitCount(bb & mask);
}

/**
 * resets the lsb in the given number and returns the result.
 * @param number
 * @return
 */
inline BB lsbReset(BB number) { return number & (number - 1); }

/**
 * find fully set groups of 4
 */
inline BB highlightGroupsOf4(BB bb) {
    bb &= (bb >> 1);
    bb &= (bb >> 2);
    return bb;
}

/**
 * stream bits of 4
 */
template<uint8_t values>
constexpr inline BB repeatGroupsOf4() {
    BB bb {};
    bb |= (BB) values & 0xF;
    bb |= (bb << 32);
    bb |= (bb << 16);
    bb |= (bb << 8);
    bb |= (bb << 4);
    return bb;
}

/**
 * prints the given bitboard as a bitmap to the standard output stream
 * @param bb
 */
inline void printBitboard(BB bb) {
    for (int i = 7; i >= 0; i--) {
        for (int n = 0; n < 8; n++) {
            if ((bb >> (i * 8 + n)) & (BB) 1) {
                std::cout << "1";
            } else {
                std::cout << "0";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

/**
 * prints the given bits starting at the msb and ending at the lsb
 * @param bb
 */
inline void printBits(BB bb) {
    for (int i = 63; i >= 0; i--) {
        if (getBit(bb, i))
            std::cout << "1";
        else
            std::cout << "0";

        if (i % 8 == 0)
            std::cout << " ";
    }
    std::cout << "\n";
}

/**
 *
 */
template<unsigned N, typename T = BB>
inline T mask() {
    return (T) (((T) 1 << N) - 1);
}

#endif    // BINARYPOSITIONWRAPPER__BITBOARD_H_
