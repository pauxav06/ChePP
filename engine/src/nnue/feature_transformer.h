#ifndef CHEPP_FEATURE_TRANSFORMER_H
#define CHEPP_FEATURE_TRANSFORMER_H

#include "utils.h"
#include <utility>

namespace chepp::nnue {
    struct FeatureTransformer {
        static constexpr auto MaxChanges = 32;
        using FeatureT                   = uint16_t;
        using RetT                       = ArrayStack<FeatureT, MaxChanges>;

        static constexpr size_t n_features_v = 32 * 11 * 64;

        static bool
        needs_refresh(const Position& prev, const Position& cur, const Color view) {
            return prev.ksq(view) != cur.ksq(view);
        }

        static RetT
        get_refresh_features(const Position& pos, const Color side) {
            RetT res{};
            for (const Square sq : pos.occupancy()) {
                res.push_back(get_index(side, pos.ksq(side), sq, pos.piece_at(sq)));
            }
            return res;
        }

        static std::pair<RetT, RetT>
        get_incremental_features(const Position& prev, const Position& cur, const Color side) {
            std::pair<RetT, RetT> res{};
            for (const auto color : Color::all()) {
                for (const auto sq : prev.occupancy(color) ^ cur.occupancy(color)) {
                    if (prev.occupancy(color).is_set(sq)) {
                        res.second.push_back(get_index(side, prev.ksq(side), sq, prev.piece_at(sq)));
                    } else {
                        res.first.push_back(get_index(side, cur.ksq(side), sq, cur.piece_at(sq)));
                    }
                }
            }
            return res;
        }

      private:
        static int
        king_square_index(Square ksq) {
            static EnumArray<FeatureT, Square> WKSqH = {0,  1,  2,  3,  3,  2,  1,  0,  4,  5,  6,  7,  7,  6,  5,  4,
                                                        8,  9,  10, 11, 11, 10, 9,  8,  12, 13, 14, 15, 15, 14, 13, 12,
                                                        16, 17, 18, 19, 19, 18, 17, 16, 20, 21, 22, 23, 23, 22, 21, 20,
                                                        24, 25, 26, 27, 27, 26, 25, 24, 28, 29, 30, 31, 31, 30, 29, 28};

            return WKSqH[ksq];
        }

        static FeatureT
        get_index(Color view, Square king_square, Square piece_square, Piece piece) {
            auto relative_piece_square = (view == WHITE ? piece_square : piece_square.flipped_horizontally());
            auto relative_king_square  = (view == WHITE ? king_square : king_square.flipped_horizontally());
            if (king_square.file() > FILE_D) {
                relative_piece_square = relative_piece_square.flipped_vertically();
            }
            int  piece_idx = piece.type() == KING ? 0 : 1 + piece.type().value() * 2 + (piece.color() == view ? 0 : 1);
            auto res{king_square_index(relative_king_square) + relative_piece_square.value() * 32 +
                     piece_idx * 32 * 64};
            assert(res >= 0 && res <= std::numeric_limits<FeatureT>::max());
            return static_cast<FeatureT>(res);
        }
    };
} // namespace chepp::nnue

#endif // CHEPP_FEATURE_TRANSFORMER_H
