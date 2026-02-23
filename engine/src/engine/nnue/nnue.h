#ifndef CHEPP_LAYERS_H
#define CHEPP_LAYERS_H

#include "accumulator.h"
#include "affine.h"
#include "layer_base.h"
#include "relu.h"

namespace chepp::nnue {
    template <typename Layer, auto... Configs>
    struct LayerConfig {
        using layer_t                 = Layer;
        static constexpr auto configs = std::make_tuple(Configs...);
    };

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
            static EnumArray<int, Square> WKSqH = {0,  1,  2,  3,  3,  2,  1,  0,  4,  5,  6,  7,  7,  6,  5,  4,
                                                   8,  9,  10, 11, 11, 10, 9,  8,  12, 13, 14, 15, 15, 14, 13, 12,
                                                   16, 17, 18, 19, 19, 18, 17, 16, 20, 21, 22, 23, 23, 22, 21, 20,
                                                   24, 25, 26, 27, 27, 26, 25, 24, 28, 29, 30, 31, 31, 30, 29, 28};

            return WKSqH[ksq];
        }

        static int
        get_index(Color view, Square king_square, Square piece_square, Piece piece) {
            auto relative_piece_square = (view == WHITE ? piece_square : piece_square.flipped_horizontally());
            auto relative_king_square  = (view == WHITE ? king_square : king_square.flipped_horizontally());
            if (king_square.file() > FILE_D) {
                relative_piece_square = relative_piece_square.flipped_vertically();
            }
            int piece_idx = piece.type() == KING ? 0 : 1 + piece.type().value() * 2 + (piece.color() == view ? 0 : 1);
            return king_square_index(relative_king_square) + relative_piece_square.value() * 32 + piece_idx * 32 * 64;
        }
    };

    struct BigArch {
        static constexpr std::size_t buckets = 8;
        using accum_t                        = AccumulatorLayer<uint16_t, 22528, int16_t, 1024>;
        using psqt_t                         = AccumulatorLayer<uint16_t, 22528, int32_t, buckets>;
        using act0_t                         = ClippedReLULayer<int16_t, 1024, uint8_t, 1>;
        using l1_t                           = AffineLayer<uint8_t, 2048, int32_t, 16, int8_t, int32_t>;
        using act1_t                         = ClippedReLULayer<int32_t, 16, uint8_t, 64>;
        using l2_t                           = AffineLayer<uint8_t, 16, int32_t, 32, int8_t, int32_t>;
        using act2_t                         = ClippedReLULayer<int32_t, 32, uint8_t, 64>;
        using l3_t                           = AffineLayer<uint8_t, 32, int32_t, 1, int8_t, int32_t>;

        // clangd-format off
        using layers = std::tuple<LayerConfig<accum_t,
                                              default_config,
                                              AccumulatorSimd{1},
                                              AccumulatorSimd{2},
                                              AccumulatorSimd{4},
                                              AccumulatorSimd{8},
                                              AccumulatorSimd{16}>,
                                  LayerConfig<psqt_t, default_config>,
                                  LayerConfig<act0_t,
                                              default_config,
                                              ClippedReluSimd{1},
                                              ClippedReluSimd{2},
                                              ClippedReluSimd{4},
                                              ClippedReluSimd{8},
                                              ClippedReluSimd{16}>,
                                  LayerConfig<l1_t,
                                              default_config,
                                              AffineSimdRowMaj{1, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{2, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{4, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{8, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{1, AffineOperation::MulPairwiseAdd},
                                              AffineSimdRowMaj{2, AffineOperation::MulPairwiseAdd},
                                              AffineSimdRowMaj{4, AffineOperation::MulPairwiseAdd},
                                              AffineSimdRowMaj{8, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>,
                                  LayerConfig<act1_t, default_config>,
                                  LayerConfig<l2_t,
                                              default_config,
                                              AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>,
                                  LayerConfig<act2_t, default_config>,
                                  LayerConfig<l3_t, default_config>>;
        // clangd-format on
    };

    // This variable controls which layers will be compiled
    static constexpr auto ALL_LAYERS = std::tuple_cat(BigArch::layers{});
    void
    register_all_layers(KernelRegistry&);
} // namespace chepp::nnue

#endif // CHEPP_LAYERS_H
