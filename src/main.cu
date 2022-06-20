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

#include "misc/config.h"
#include "archs/Koivisto.h"

#include "trainer.h"

#include <iostream>
#include <vector>

using namespace std;

int main() {
    init();

    const string   data_path = R"(H:\Koivisto Resourcen\Training Data 8.9\noob\)";
    const string   output    = "../resources/runs/new_system_1/";

    vector<string> files {};
    for (int i = 1; i <= 192; i++)
        files.push_back(data_path + "shuffled_" + to_string(i) + ".bin");

    Trainer<Koivisto> trainer {};
    trainer.fit(
        files,
        vector<string> {R"(H:\Koivisto Resourcen\Training Data 7.9\reshuffled_generated_0.txt.bin)"},
        output);

    //        auto layers = ChessDotCpp::get_layers();
    //        Network network{layers};
    //        network.loadWeights(output + "weights-epoch20.nnue");
    //        BatchLoader batch_loader{files, 16384};
    //        batch_loader.start();
    //        test_fen<ChessDotCpp>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w QKqk - 0
    //        1", ChessDotCpp::Inputs);
    //    test_fen<ChessDotCpp>(network, "8/2kpn3/2p1b3/1p4P1/5P2/4PQ2/3PK3/8 w - - 0 1",
    //    ChessDotCpp::Inputs); computeScalars<ChessDotCpp>(batch_loader, network, 1024,
    //    ChessDotCpp::Inputs); quantitize_shallow(output + "20.net", network, 128, 256);

    close();
}
