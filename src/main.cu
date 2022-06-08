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

#include "archs/Berserk.h"
#include "misc/config.h"
#include "trainer.h"
#include "archs/Koivisto.h"

#include <iostream>
#include <vector>

using namespace std;

int main() {
    init();


    const string data_path = R"(H:\Koivisto Resourcen\Training Data 8.9\noob\)";
    const string output    = "../resources/runs/experiment_28/";

    vector<string> files {};
    for (int i = 1; i <= 192; i++)
        files.push_back(data_path + "shuffled_" + to_string(i) + ".bin");

    Trainer<Koivisto> trainer {};
    trainer.fit(files, vector<string> {R"(H:\Koivisto Resourcen\Training Data 7.9\reshuffled_generated_0.txt.bin)"}, output);

//    auto layers = Koivisto::get_layers();
//    Network network{layers};
//    network.loadWeights(output + "weights-epoch380.nnue");
//    BatchLoader batch_loader{files, 16384};
//    batch_loader.start();
//    quantitize_shallow(output + "380.net", network, 32, 128);
//    computeScalars<Koivisto>(batch_loader, network, 1024, 12288);

    close();
}
