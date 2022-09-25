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

#ifndef CUDAD_SRC_MAPPINGS_BERSERK_H_
#define CUDAD_SRC_MAPPINGS_BERSERK_H_

#include "../activations/ReLU.h"
#include "../activations/Sigmoid.h"
#include "../data/SArray.h"
#include "../data/SparseInput.h"
#include "../dataset/dataset.h"
#include "../layer/DenseLayer.h"
#include "../layer/DuplicateDenseLayer.h"
#include "../loss/Loss.h"
#include "../loss/MPE.h"
#include "../optimizer/Adam.h"
#include "../optimizer/Optimiser.h"
#include "../position/fenparsing.h"

#include <tuple>

class Berserk {

    public:
    static constexpr int   Inputs        = 16 * 12 * 64;
    static constexpr int   L2            = 512;
    static constexpr int   Outputs       = 1;
    static constexpr float SigmoidScalar = 1.0 / 160;

    static Optimiser*      get_optimiser() {
             Adam* optim     = new Adam();
             optim->lr       = 1e-2;
             optim->beta1    = 0.95;
             optim->beta2    = 0.999;
             optim->schedule = LRScheduler(250, 0.1);

             return optim;
    }

    static Loss* get_loss_function() {
        MPE* loss_f = new MPE(2.5, false);

        return loss_f;
    }

    static std::vector<LayerInterface*> get_layers() {
        DuplicateDenseLayer<Inputs, L2, ReLU>* l1 = new DuplicateDenseLayer<Inputs, L2, ReLU>();
        l1->lasso_regularization                  = 1.0 / 3355443.2;

        DenseLayer<L2 * 2, Outputs, Sigmoid>* l2  = new DenseLayer<L2 * 2, Outputs, Sigmoid>();
        dynamic_cast<Sigmoid*>(l2->getActivationFunction())->scalar = SigmoidScalar;

        return std::vector<LayerInterface*> {l1, l2};
    }

    static void assign_inputs_batch(DataSet&       positions,
                                    SparseInput&   i0,
                                    SparseInput&   i1,
                                    SArray<float>& output,
                                    SArray<bool>&  output_mask) {
        i0.clear();
        i1.clear();
        output_mask.clear();

#pragma omp parallel for schedule(static) num_threads(8)
        for (int i = 0; i < positions.positions.size(); i++)
            assign_input(positions.positions[i], i0, i1, output, output_mask, i);
    }

    static int king_square_index(int relative_king_square) {
        constexpr int indices[N_SQUARES] {
            -1, -1, -1, -1, 14, 14, 15, 15,    //
            -1, -1, -1, -1, 14, 14, 15, 15,    //
            -1, -1, -1, -1, 12, 12, 13, 13,    //
            -1, -1, -1, -1, 12, 12, 13, 13,    //
            -1, -1, -1, -1, 8,  9,  10, 11,    //
            -1, -1, -1, -1, 8,  9,  10, 11,    //
            -1, -1, -1, -1, 4,  5,  6,  7,     //
            -1, -1, -1, -1, 0,  1,  2,  3,     //
        };

        return indices[relative_king_square];
    }

    static int index(Square psq, Piece p, Square kingSquare, Color view) {
        const PieceType pieceType  = getPieceType(p);
        const Color     pieceColor = getPieceColor(p);

        psq ^= 56;
        kingSquare ^= 56;

        const int oP  = pieceType + 6 * (pieceColor != view);
        const int oK  = (7 * !(kingSquare & 4)) ^ (56 * view) ^ kingSquare;
        const int oSq = (7 * !(kingSquare & 4)) ^ (56 * view) ^ psq;

        return king_square_index(oK) * 12 * 64 + oP * 64 + oSq;
    }

    static void assign_input(Position&      p,
                             SparseInput&   i0,
                             SparseInput&   i1,
                             SArray<float>& output,
                             SArray<bool>&  output_mask,
                             int            id) {

        // track king squares
        Square wKingSq = p.getKingSquare<WHITE>();
        Square bKingSq = p.getKingSquare<BLACK>();

        BB     bb {p.m_occupancy};
        int    idx = 0;

        while (bb) {
            Square sq                    = bitscanForward(bb);
            Piece  pc                    = p.m_pieces.getPiece(idx);

            auto   piece_index_white_pov = index(sq, pc, wKingSq, WHITE);
            auto   piece_index_black_pov = index(sq, pc, bKingSq, BLACK);

            if (p.m_meta.getActivePlayer() == WHITE) {
                i0.set(id, piece_index_white_pov);
                i1.set(id, piece_index_black_pov);
            } else {
                i1.set(id, piece_index_white_pov);
                i0.set(id, piece_index_black_pov);
            }

            bb = lsbReset(bb);
            idx++;
        }

        float p_value = p.m_result.score;
        float w_value = p.m_result.wdl;

        // flip if black is to move -> relative network style
        if (p.m_meta.getActivePlayer() == BLACK) {
            p_value = -p_value;
            w_value = -w_value;
        }

        float p_target  = 1 / (1 + expf(-p_value * SigmoidScalar));
        float w_target  = (w_value + 1) / 2.0f;

        output(id)      = (p_target + w_target) / 2;
        output_mask(id) = true;
    }
};

#endif
