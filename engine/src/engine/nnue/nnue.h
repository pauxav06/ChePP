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
