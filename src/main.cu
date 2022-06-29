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

#include "archs/Koivisto.h"
#include "misc/config.h"
#include "numerical/finite_difference.h"
#include "trainer.h"
#include "quantitize.h"

#include <iostream>
#include <vector>

using namespace std;

int main() {
    init();

    const string   data_path = R"(D:\Koivisto Resourcen\Training Data 8.9\noob\)";
    const string   output    = R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\CudAD\resources\runs\experiment_36\)";

    vector<string> files {};
    for (int i = 1; i <= 192; i++)
        files.push_back(data_path + "shuffled_" + to_string(i) + ".bin");

    Trainer<Koivisto> trainer {};
    trainer.fit(
        files,
        vector<string> {R"(D:\Koivisto Resourcen\Training Data 7.9\reshuffled_generated_0.txt.bin)"},
        output);

//    auto layers = Koivisto::get_layers();
//    Network network{std::get<0>(layers),std::get<1>(layers)};
//    network.setLossFunction(Koivisto::get_loss_function());
//    network.loadWeights(output + "weights-epoch10.nnue");
//

//    test_fen<Koivisto>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch10.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch20.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch30.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch40.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch50.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch60.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch70.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch80.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch90.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch100.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch110.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch120.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch130.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch140.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch160.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch10.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//
//    DenseMatrix target{1,1};
//    target(0,0) = 0.3;
//    SArray<bool> target_mask{1};
//    target_mask.malloc_cpu();
//    target_mask.malloc_gpu();
//    target_mask(0) = true;
//    target_mask.gpu_upload();
//    target.gpu_upload();
//
//    finite_difference(network, target, target_mask);

//    network.loadWeights(output + "weights-epoch120.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//    network.loadWeights(output + "weights-epoch30.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//
//    network.loadWeights(output + "weights-epoch50.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//
//
//    network.loadWeights(output + "weights-epoch80.nnue");
//    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");

//    test_fen<Koivisto>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    test_fen<Koivisto>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

//
//    BatchLoader batch_loader{files, 16384};
//    batch_loader.start();
//    computeScalars<Koivisto>(batch_loader, network, 128);
//
//    auto f = openFile(output + "200.net");
//    writeLayer<int16_t, int16_t>(f, network, 0, 32, 32);
//    writeLayer<int16_t, int32_t>(f, network, 4, 128, 128 * 32);
//    closeFile(f);

    close();
}
