//
// Created by paul on 7/30/25.
//

#include <ChePP/engine/movegen.h>
#include <gtest/gtest.h>

TEST(ThreeFoldRepetitions, FourNullMoveIsDraw)
{

    Positions positions{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(positions.is_repetition(), false);
        positions.do_move(Move::null());

    }
    ASSERT_EQ(positions.is_repetition(), true);
}

TEST(ThreeFoldRepetitions, TwoNightShuffleIsDraw)
{

    Positions positions{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
    positions.do_move(Move::make<NORMAL>(G1, H3));
    positions.do_move(Move::make<NORMAL>(G8, H6));

    for (int i = 0; i < 2; i++)
    {
        ASSERT_EQ(positions.is_repetition(), false) << " at iterations nb: " << i;
        positions.do_move(Move::make<NORMAL>(B1, C3));
        positions.do_move(Move::make<NORMAL>(B8, C6));
        positions.do_move(Move::make<NORMAL>(C3, B1));
        positions.do_move(Move::make<NORMAL>(C6, B8));
    }
    ASSERT_EQ(positions.is_repetition(), true);
}

TEST(FiftyMoveRule, FiftyRookShuffleIsDraw)
{
    Positions positions("K1k5/pppppppp/8/8/8/8/8/R7 w KQkq - 0 1");

    Square sq_from = A1;
    for (Square sq_to = B1; sq_to <= B1 + 24; ++sq_to)
    {
        const Move mv = Move::make<NORMAL>(sq_from, sq_to);
        positions.do_move(mv);
        positions.do_move(Move::null());
        positions.do_move(Move::null());
        positions.do_move(Move::null());

        sq_from = sq_to;

        ASSERT_EQ(positions.is_repetition(), sq_to == (B1 + 24)) << "halfmoves clock: " << positions.last().halfmove_clock();
    }


}

TEST(FiftyMoveRule, PawnMoveResetsFiftyRookShuffleIsDraw)
{
    Positions positions("K1k5/pppppppp/8/8/8/8/8/R7 w KQkq - 0 1");

    Square sq_from = A1;
    for (Square sq_to = B1; sq_to < B1 + 25; ++sq_to)
    {
        const Move mv = Move::make<NORMAL>(sq_from, sq_to);
        if (sq_from == A1 + 10)
        {
            positions.do_move(Move::null());
            positions.do_move(Move::make<NORMAL>(H7, H6));
        }
        positions.do_move(mv);
        positions.do_move(Move::null());
        positions.do_move(Move::null());
        positions.do_move(Move::null());

        sq_from = sq_to;

        ASSERT_EQ(positions.is_repetition(), false) << "halfmoves clock: " << positions.last().halfmove_clock();
    }

}