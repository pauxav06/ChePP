//
// Created by paul on 10/2/25.
//
#include <ChePP/engine/nnue.h>
#include <ChePP/engine/init.h>
#include <gtest/gtest.h>

#include <random>
#include <algorithm>
#include <ranges>



TEST(AccumTest, IncrIsSameAsRefresh) {
    Positions positions;
    positions.set_fen(start_fen);
    nnue::Accumulators accumulators{positions.last()};
    int total = 0;

    auto accum_perft = [&] (auto self, int depth) -> void
    {
        if (depth <= 0) return;
        auto moves = gen_legal(positions.last());
        for (const auto& move : moves)
        {
            positions.do_move(move);
            accumulators.do_move(positions[positions.ply() - 1], positions.last());
            total++;
            const auto side = positions.last().side_to_move();
            ASSERT_EQ(
                nnue::network.evaluate(accumulators.last(), side),
                nnue::network.evaluate(nnue::Accumulator{positions.last()}, side)
                ) << "Incoherent evaluation for\n" << positions.last() << "at depth " << positions.ply()
                << " / " << positions.ply() + depth - 1 << std::endl;
            self(self, depth - 1);
            accumulators.undo_move();
            positions.undo_move();
        }
    };

    auto launch_with_fen = [&] (const std::string& fen)
    {
        positions.set_fen(fen);
        accumulators = nnue::Accumulators{positions.last()};
        accum_perft(accum_perft, 3);
    };

    launch_with_fen(start_fen);
    launch_with_fen("8/3K4/2p5/p2b2r1/5k2/8/8/1q6 b - - 1 67");
    launch_with_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    launch_with_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    std::cout << "tested " << total <<" nnue positions" << std::endl;
}
