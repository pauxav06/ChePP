
#include "ChePP/engine/bitboard.h"
#include "ChePP/engine/types.h"
#include "ChePP/engine/position.h"
#include "ChePP/engine/movegen.h"
#include "ChePP/engine/search.h"
#include "ChePP/engine/UCI.h"
#include <chrono>
#include <iostream>
#include <random>

#include <iostream>

#include "ChePP/engine/types.h"
#include "ChePP/engine/nnue.h"



void perft_divide(const Position& prev, int depth) {
    MoveList l = gen_moves(prev);
    size_t total = 0;


    for (auto [mv, _] : l)
    {
        Position next = prev;
        next.do_move(mv);

        size_t nodes = 0;
        perft(next, depth - 1, nodes);


        std::cout << next.piece_at(mv.from_sq()) << " " << mv.from_sq() << " " << mv.to_sq() << ": " << nodes << '\n';
        total += nodes;
    }


    std::cout << "Total: " << total << '\n';
}


/**
void init_random_weights(layer_t<int16_t, feature_t::n_features, nnue_t::s_accumulator_size>& layer, int16_t seed = 1) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int16_t> dist(1, 3);

    for (auto& row : layer.m_weights) {
        for (auto& w : row) {
            w = dist(rng);
        }
    }

    for (auto& b : layer.m_biases) {
        b = dist(rng);
    }
}

void print_accumulator(accumulator_t<nnue_t::s_accumulator_size>& acc, color_t c) {
    std::cout << (c == WHITE ? "WHITE accumulator:\n" : "BLACK accumulator:\n");
    for (size_t i = 0; i < nnue_t::s_accumulator_size; ++i) {
        std::cout << acc[c][i] << " ";
        if (i % 16 == 15) std::cout << "\n";
    }
    std::cout << "\n";
}
*
*
 */


int main() {
    g_tt.init(512);

    UCIEngine engine{};
    engine.loop();

    return 0;
}


