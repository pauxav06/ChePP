//
// Created by paul on 10/13/25.
//

#include "core.h"

#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace std;

namespace chepp {
    TEST(ArrayStackTest, BasicOperations) {
        ArrayStack<int, 4> s;
        EXPECT_TRUE(s.empty());
        EXPECT_EQ(s.size(), 0u);
        s.push_back(10);
        s.push_back(20);
        s.push_back(30);
        EXPECT_EQ(s.size(), 3u);
        EXPECT_EQ(s[0], 10);
        EXPECT_EQ(s[1], 20);
        EXPECT_EQ(s.front(), 10);
        EXPECT_EQ(s.back(), 30);

        s.shrink(1);
        EXPECT_EQ(s.size(), 2u);
        EXPECT_EQ(s.back(), 20);
        s.clear();
        EXPECT_TRUE(s.empty());
    }

    template <typename Enum>
    class EnumStringTest : public ::testing::Test {};

    using MyTypes = ::testing::Types<File, Rank, Square, PieceType, Color, Piece, CastlingType>;

    TYPED_TEST_SUITE(EnumStringTest, MyTypes);

    TYPED_TEST(EnumStringTest, RoundtripConversion) {
        for (auto e : TypeParam::all()) {
            std::string s    = e.to_string();
            auto        back = TypeParam::from_string(s);
            EXPECT_TRUE(back.has_value()) << "Empty optional for valid enum member: " << s;
            EXPECT_EQ(*back, e) << "Forward and back do not match: " << s << " -> " << *back;
        }
    }

    std::string
    random_string(const size_t length) {
        const std::string chars = "abcdefghijklmnopqrstuvwxyz"
                                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                  "0123456789";

        thread_local std::random_device rd;
        thread_local std::mt19937       gen(rd());
        std::uniform_int_distribution<> dis(0, chars.size() - 1);

        std::string result;
        result.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            result += chars[dis(gen)];
        }
        return result;
    }

    TYPED_TEST(EnumStringTest, InjectiveAndOptionalIfInvalid) {
        std::unordered_map<std::string, TypeParam> map;
        const size_t                               max_length = []() {
            size_t ret = 0;
            for (const TypeParam e : TypeParam::all()) {
                ret = std::max(ret, e.to_string().size());
            }
            return ret;
        }();

        thread_local std::random_device rd;
        thread_local std::mt19937       gen(rd());
        std::uniform_int_distribution<> dis(0, max_length + 1);

        constexpr size_t iter = 10000;

        for (size_t i = 0; i < iter; i++) {
            const std::string                          str    = random_string(dis(gen));
            const tl::expected<TypeParam, std::string> e      = TypeParam::from_string(str);
            const bool                                 in_map = map.contains(str);
            if (e.has_value()) {
                if (in_map) {
                    EXPECT_TRUE(e.has_value());
                    EXPECT_EQ(map.at(str), e);
                }
                map.insert_or_assign(str, *e);
            } else {
                EXPECT_FALSE(in_map);
            }
        }
        EXPECT_LE(map.size(), TypeParam::total()) << "More enums parsed than valid options";
    }

    TEST(EnumBaseTest, FileRankSquareConversions) {
        for (const File f : File::all()) {
            for (const Rank r : Rank::all()) {
                Square sq{f, r};
                EXPECT_EQ(sq.file(), f);
                EXPECT_EQ(sq.rank(), r);
                EXPECT_EQ(sq.to_string(), (f.to_string() + r.to_string()));
                EXPECT_EQ(sq.flipped_vertically().rank(), r);
                EXPECT_EQ(sq.flipped_vertically().file() + sq.file(), FILE_H);
                EXPECT_EQ(sq.flipped_horizontally().file(), f);
                EXPECT_EQ(sq.flipped_horizontally().rank() + sq.rank(), RANK_8);
            }
        }
    }

    TEST(EnumArrayTest, SingleEnumFillPred) {
        EnumArray<int, File> arr{};
        arr.fill_pred([](File f) { return static_cast<int>(f.index()); });
        EXPECT_EQ(arr[FILE_A], 0);
        EXPECT_EQ(arr[FILE_C], 2);
    }

    TEST(EnumArrayTest, MultiEnumFillPred) {
        EnumArray<int, File, Rank> arr{};
        arr.fill_pred([](const File f, const Rank r) { return static_cast<int>(f.index() * 10 + r.index()); });
        EXPECT_EQ(arr[FILE_B][RANK_3], 1 * 10 + 2);
    }

    TEST(PieceTest, PieceTypeAndValues) {
        for (const Piece pc : Piece::all()) {
            const auto back = Piece{pc.color(), pc.type()};
            EXPECT_EQ(pc, back);
        }
    }

    TEST(ColorTest, Opposite) {
        EXPECT_EQ(~WHITE, BLACK);
        EXPECT_EQ(~(~BLACK), BLACK);
        // EXPECT_EQ(~NO_COLOR, NO_COLOR);
    }

    TEST(CastlingTest, KingAndRookMoves) {
        auto [kfrom, kto] = WHITE_KINGSIDE.king_move();
        auto [rfrom, rto] = WHITE_KINGSIDE.rook_move();
        EXPECT_EQ(kfrom.to_string(), "e1");
        EXPECT_EQ(kto.to_string(), "g1");
        EXPECT_EQ(rfrom.to_string(), "h1");
        EXPECT_EQ(rto.to_string(), "f1");
    }

    TEST(CastlingRightsTest, LostFromMove) {
        CastlingRights cr = CastlingRights::all();
        Move           rookMove{H1, F1};
        CastlingRights lost = cr.lost_from_move(rookMove);
        EXPECT_TRUE(lost.has(WHITE_KINGSIDE));
        EXPECT_FALSE(lost.has(WHITE_QUEENSIDE));
    }

    TEST(BitHelpersTest, PopcountAndLSBMSB) {
        using u64 = uint64_t;
        u64 x     = (u64(1) << 0) | (u64(1) << 3) | (u64(1) << 63);
        EXPECT_EQ(bit::popcount(x), 3);
        EXPECT_EQ(bit::get_lsb(x), 0);
        EXPECT_EQ(bit::get_msb(x), 63);

        u64 y   = x;
        int lsb = bit::pop_lsb(y);
        EXPECT_EQ(lsb, 0);
        EXPECT_EQ(bit::get_lsb(y), 3);

        u64 z   = x;
        int msb = bit::pop_msb(z);
        EXPECT_EQ(msb, 63);
    }

    TEST(MoveTest, BasicEncodeDecode) {
        Move m{E2, E4};
        EXPECT_EQ(m.from_sq().to_string(), "e2");
        EXPECT_EQ(m.to_sq().to_string(), "e4");
        EXPECT_EQ(m.to_string(), "e2e4");
        EXPECT_EQ(m.type_of(), NORMAL);
    }

    TEST(MoveTest, PromotionAndPromotionType) {
        Move pm = Move::make<PROMOTION>(E7, E8, QUEEN);
        EXPECT_EQ(pm.type_of(), PROMOTION);
        EXPECT_EQ(pm.promotion_type(), QUEEN);
        EXPECT_EQ(pm.to_string(), "e7e8q");
        Move pm1 = Move::make<PROMOTION>(F2, F1, KNIGHT);
        EXPECT_EQ(pm1.type_of(), PROMOTION);
        EXPECT_EQ(pm1.promotion_type(), KNIGHT);
        EXPECT_EQ(pm1.to_string(), "f2f1n");
    }

    TEST(MoveTest, CastlingEncoding) {
        Move cm = Move::make<CASTLING>(E1, G1, WHITE_KINGSIDE);
        EXPECT_EQ(cm.type_of(), CASTLING);
        EXPECT_EQ(cm.castling_type(), WHITE_KINGSIDE);
    }

    TEST(MoveFromUciTest, NormalPawnMove) {
        EnumArray<Piece, Square> pieces{};
        pieces.fill(NO_PIECE);
        pieces[E2] = W_PAWN;
        Move::UCICtx info{pieces, NO_SQUARE, CastlingRights::none()};
        auto         mv = Move::from_uci("e2e4", info);
        ASSERT_TRUE(mv.has_value());
        EXPECT_EQ(mv->from_sq().to_string(), "e2");
        EXPECT_EQ(mv->to_sq().to_string(), "e4");
    }

    TEST(MoveFromUciTest, EnPassant) {
        EnumArray<Piece, Square> pieces{};
        pieces.fill(NO_PIECE);
        pieces[E5] = W_PAWN;
        Move::UCICtx info{pieces, D6, CastlingRights::none()};
        auto         mv = Move::from_uci("e5d6", info);
        ASSERT_TRUE(mv.has_value());
        EXPECT_EQ(mv->type_of(), EN_PASSANT);

        Move::UCICtx info1{pieces, NO_SQUARE, CastlingRights::none()};
        auto         mv1 = Move::from_uci("e5d6", info1);
        ASSERT_TRUE(mv1.has_value());
        EXPECT_EQ(mv1->type_of(), NORMAL);
    }

    TEST(MoveFromUciTest, Castling) {
        EnumArray<Piece, Square> pieces{};
        pieces.fill(NO_PIECE);
        pieces[E1] = W_KING;
        pieces[H1] = W_ROOK;
        CastlingRights cr{WHITE_KINGSIDE};
        Move::UCICtx   info{pieces, NO_SQUARE, cr};
        auto           mv = Move::from_uci("e1g1", info);
        ASSERT_TRUE(mv.has_value());
        EXPECT_EQ(mv->type_of(), CASTLING);
        EXPECT_EQ(mv->castling_type(), WHITE_KINGSIDE);

        cr.remove(CastlingRights::all());

        Move::UCICtx info1{pieces, NO_SQUARE, cr};
        auto         mv1 = Move::from_uci("e1g1", info1);
        ASSERT_TRUE(mv1.has_value());
        EXPECT_EQ(mv1->type_of(), NORMAL);
    }

    TEST(CastlingRights, RoundtripAndInvalid) {
        EXPECT_EQ(std::string(CASTLING_NONE.to_string()), std::string("-"));
        EXPECT_EQ(std::string(CASTLING_K.to_string()), std::string("K"));
        EXPECT_EQ(std::string(CASTLING_Q.to_string()), std::string("Q"));
        EXPECT_EQ(std::string(CASTLING_KQ.to_string()), std::string("KQ"));
        EXPECT_EQ(std::string(CASTLING_kq.to_string()), std::string("kq"));
        EXPECT_EQ(std::string(CASTLING_KQkq.to_string()), std::string("KQkq"));

        auto opt = CastlingRights::from_string("KQ");
        ASSERT_TRUE(opt.has_value());
        EXPECT_EQ(opt->mask(), CASTLING_KQ.mask());

        auto opt2 = CastlingRights::from_string("KQkq");
        ASSERT_TRUE(opt2.has_value());
        EXPECT_EQ(opt2->mask(), CASTLING_KQkq.mask());

        auto opt_bad = CastlingRights::from_string("invalid");
        EXPECT_FALSE(opt_bad.has_value());
    }

    TEST(CastlingRights, AddThenRemove) {
        CastlingRights cr = CastlingRights::none();
        EXPECT_TRUE(cr.empty());

        cr.add(WHITE_KINGSIDE);
        EXPECT_TRUE(cr.has(WHITE_KINGSIDE));
        EXPECT_FALSE(cr.has(WHITE_QUEENSIDE));
        EXPECT_EQ(cr.to_string(), std::string("K"));

        cr.add(WHITE_QUEENSIDE);
        EXPECT_TRUE(cr.has(WHITE_QUEENSIDE));
        EXPECT_EQ(cr.to_string(), std::string("KQ"));

        cr.remove(WHITE_KINGSIDE);
        EXPECT_FALSE(cr.has(WHITE_KINGSIDE));
        EXPECT_TRUE(cr.has(WHITE_QUEENSIDE));
        EXPECT_EQ(cr.to_string(), std::string("Q"));
    }

    TEST(CastlingRights, RemoveOtherAndKeep) {
        CastlingRights all = CastlingRights::all();
        EXPECT_EQ(all.mask(), CASTLING_KQkq.mask());

        all.remove(CASTLING_KQ);
        EXPECT_EQ(all.mask(), CASTLING_kq.mask());
        EXPECT_FALSE(all.has_any_color(WHITE));
        EXPECT_TRUE(all.has_any_color(BLACK));

        CastlingRights fresh = CastlingRights::all();
        fresh.keep(CASTLING_K);
        EXPECT_EQ(fresh.mask(), CASTLING_K.mask());
        EXPECT_TRUE(fresh.has(WHITE_KINGSIDE));
        EXPECT_FALSE(fresh.has(WHITE_QUEENSIDE));
        EXPECT_FALSE(fresh.has(BLACK_KINGSIDE));
    }

    TEST(CastlingRights, IntConstructorBounds) {
        CastlingRights big(static_cast<int>(0xFFFF));
        EXPECT_EQ(big.mask(), CASTLING_KQkq.mask());

        CastlingRights small(3);
        EXPECT_EQ(small.mask(), CASTLING_KQ.mask());
    }

    TEST(CastlingRights, Queries) {
        CastlingRights cr{WHITE_KINGSIDE, BLACK_QUEENSIDE};
        EXPECT_TRUE(cr.has_any());
        EXPECT_TRUE(cr.has(WHITE_KINGSIDE));
        EXPECT_TRUE(cr.has(BLACK_QUEENSIDE));
        EXPECT_FALSE(cr.has(WHITE_QUEENSIDE));
        EXPECT_TRUE(cr.has_any_color(WHITE));
        EXPECT_TRUE(cr.has_any_color(BLACK));
    }

    TEST(CastlingRights, KingAndRookMovesRemoveCorrectRights) {
        CastlingRights full = CastlingRights::all();
        EXPECT_EQ(full.mask(), CASTLING_KQkq.mask());

        Move           mk   = Move::make<NORMAL>(E1, G1);
        CastlingRights lost = full.lost_from_move(mk);
        EXPECT_EQ(lost.mask(), CASTLING_KQ.mask());

        CastlingRights onlyK = CASTLING_K;
        CastlingRights lost2 = onlyK.lost_from_move(mk);
        EXPECT_EQ(lost2.mask(), CASTLING_K.mask());

        Move           rook_h1 = Move::make<NORMAL>(H1, F1);
        CastlingRights lost3   = CASTLING_KQkq.lost_from_move(rook_h1);
        EXPECT_EQ(lost3.mask(), CASTLING_K.mask());

        Move           rook_a8 = Move::make<NORMAL>(A8, B8);
        CastlingRights lost4   = CASTLING_KQkq.lost_from_move(rook_a8);
        EXPECT_EQ(lost4.mask(), CASTLING_q.mask());
    }
} // namespace chepp
