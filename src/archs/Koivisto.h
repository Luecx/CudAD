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

#ifndef CUDAD_SRC_MAPPINGS_KOIVISTO_H_
#define CUDAD_SRC_MAPPINGS_KOIVISTO_H_

#include "../activations/ReLU.h"
#include "../activations/Sigmoid.h"
#include "../data/SArray.h"
#include "../data/SparseInput.h"
#include "../dataset/dataset.h"
#include "../layer/DenseLayer.h"
#include "../loss/Loss.h"
#include "../loss/MPE.h"
#include "../optimizer/Adam.h"
#include "../optimizer/Optimiser.h"
#include "../position/fenparsing.h"
#include "../layer/Input.h"
#include "../layer/ActivationLayer.h"
#include "../layer/MergeLayer.h"
#include "../network/Network.h"

#include <tuple>

class Koivisto {

    public:
    static constexpr int   Inputs        = 16 * 12 * 64;
    static constexpr int   L2            = 512;
    static constexpr int   Outputs       = 1;
    static constexpr float SigmoidScalar = 2.5 / 400;


    using LayerList = std::vector<LayerInterface*>;

    static Optimiser*      get_optimiser() {
        Adam* optim  = new Adam();
        optim->lr    = 1e-2;
        optim->beta1 = 0.95;
        optim->beta2 = 0.999;

        return optim;
    }

    static Loss* get_loss_function() {
        MPE* loss_f = new MPE(2.5, false);

        return loss_f;
    }

    static std::tuple<LayerList, LayerList> get_layers() {

        auto i1 = new Input(true, Inputs, 32);
        auto i2 = new Input(true, Inputs, 32);

        auto h1 = new DenseLayer<L2>(i1);
        auto h2 = new DenseLayer<L2>(i2, h1);

        auto m1 = new MergeLayer(h1,h2);
        auto a1 = new ActivationLayer<ReLU>(m1);

        auto h3 = new DenseLayer<1>(a1);
        auto a2 = new ActivationLayer<Sigmoid>(h3);

        a2->f.scalar = 2.5 / 400;

        return {
            {i1,i2},
            {h1,h2,m1,a1,h3,a2}
        };
    }

    static void assign_inputs_batch(DataSet&       positions,
                                    Network&       network,
                                    SArray<float>& output,
                                    SArray<bool>&  output_mask) {

        ASSERT(positions.positions.size() == in1.n);
        ASSERT(positions.positions.size() == in2.n);

        SparseInput& in1 = network.getInputs()[0]->sparse_data;
        SparseInput& in2 = network.getInputs()[1]->sparse_data;

        in1.clear();
        in2.clear();
        output_mask.clear<HOST>();

#pragma omp parallel for schedule(static) num_threads(16)
        for (int i = 0; i < positions.positions.size(); i++)
            assign_input(positions.positions[i], in1, in2, output, output_mask, i);
    }

    static int king_square_index(Square relative_king_square) {

        // clang-format off
        constexpr int indices[N_SQUARES] {
            0,  1,  2,  3,  3,  2,  1,  0,
            4,  5,  6,  7,  7,  6,  5,  4,
            8,  9,  10, 11, 11, 10, 9,  8,
            8,  9,  10, 11, 11, 10, 9,  8,
            12, 12, 13, 13, 13, 13, 12, 12,
            12, 12, 13, 13, 13, 13, 12, 12,
            14, 14, 15, 15, 15, 15, 14, 14,
            14, 14, 15, 15, 15, 15, 14, 14,
        };
        // clang-format on

        return indices[relative_king_square];
    }

    static int index(Square psq, Piece p, Square kingSquare, Color view) {
        constexpr int   pieceTypeFactor  = 64;
        constexpr int   pieceColorFactor = 64 * 6;
        constexpr int   kingSquareFactor = 64 * 6 * 2;

        const PieceType pieceType        = getPieceType(p);
        const Color     pieceColor       = getPieceColor(p);
        const Square relativeKingSquare  = view == WHITE ? kingSquare : mirrorVertically(kingSquare);
        const bool   kingSide            = fileIndex(kingSquare) > 3;
        const int    kingSquareIndex     = king_square_index(relativeKingSquare);
        Square       relativeSquare      = view == WHITE ? psq : mirrorVertically(psq);

        if (kingSide) {
            relativeSquare = mirrorHorizontally(relativeSquare);
        }

        return relativeSquare + pieceType * pieceTypeFactor + (pieceColor == view) * pieceColorFactor
               + kingSquareIndex * kingSquareFactor;
    }

    static void assign_input(Position&      p,
                             SparseInput&   in1,
                             SparseInput&   in2,
                             SArray<float>& output,
                             SArray<bool>&  output_mask,
                             int            id) {

        constexpr static float phase_values[6] {0, 1, 1, 2, 4, 0};

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
                in1.set(id, piece_index_white_pov);
                in2.set(id, piece_index_black_pov);
            } else {
                in2.set(id, piece_index_white_pov);
                in1.set(id, piece_index_black_pov);
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

        float p_target = 1 / (1 + expf(-p_value * SigmoidScalar));
        float w_target = (w_value + 1) / 2.0f;

        //    int   output_bucket = (bitCount(p.m_occupancy) - 1) / 4;

        output(id)      = (p_target + w_target) / 2;
        output_mask(id) = true;
    }
};

#endif