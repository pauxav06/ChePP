//
// Created by paul on 7/29/25.
//

#include <ChePP/engine/movegen.h>
#include <gtest/gtest.h>

struct perft_test_case_t
{
    const char*           name;
    const char*           fen;
    std::vector<uint64_t> expected_perfts;
};


inline void pft2(Positions& positions, const int ply, size_t& out)
{
    MoveList l = gen_legal(positions.last());

    if (ply == 1)
    {
        out += l.size();
        return;
    }

    for (const auto [move, score] : l)
    {
        positions.do_move(move);
        pft2(positions, ply - 1, out);
        positions.undo_move();
    }
}

TEST(EngineTest, PerftCases)
{
    const std::vector<perft_test_case_t> test_cases =
        {{"InitialPosition",
          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
          {0ULL, 20ULL, 400ULL, 8902ULL, 197281ULL, 4865609ULL /*, 119060324ULL */}},
         {"Kiwipete position 1",
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 ",
          {0ULL, 48ULL, 2039ULL, 97862ULL, 4085603ULL /*, 193690690ULL */}},
         {"Kiwipete promotions",
          "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ",
          {0ULL, 24ULL, 496ULL, 9483ULL, 182838ULL, 3605103ULL /*, 71179139ULL */}},
         {"Kiwipete position 2",
          "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
          {0ULL, 14ULL, 191ULL, 2812ULL, 43238ULL, 674624ULL, 11030083ULL /*, 178633661ULL */}},
         {"Kiwipete position 3",
          "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
          {0ULL, 6ULL, 264ULL, 9467ULL, 422333ULL, 15833292ULL /*, 706045033ULL */}},
         {"Kiwipete position 3 reversed",
          "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
          {0ULL, 6ULL, 264ULL, 9467ULL, 422333ULL, 15833292ULL /*, 706045033ULL */}},
         {"Kiwipete bonus 1", "8/3K4/2p5/p2b2r1/5k2/8/8/1q6 b - - 1 67", {0ULL, 50ULL, 279ULL}
         }
        };

    for (const auto& test_case : test_cases)
    {
        Positions pos{test_case.fen};
        for (size_t d = 1; d < test_case.expected_perfts.size(); ++d)
        {
            size_t out = 0;
            pft2(pos, d, out);
            EXPECT_EQ(out, test_case.expected_perfts[d]) << "Failed on " << test_case.name << " at depth " << d;
        }
    }
}
