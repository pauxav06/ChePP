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

        static constexpr auto layers = std::make_tuple(LayerConfig<accum_t, default_config>{},
                                                       LayerConfig<psqt_t, default_config>{},
                                                       LayerConfig<act0_t, default_config>{},
                                                       LayerConfig<l1_t, default_config>{},
                                                       LayerConfig<act1_t, default_config>{},
                                                       LayerConfig<l2_t, default_config>{},
                                                       LayerConfig<act2_t, default_config>{},
                                                       LayerConfig<l3_t, default_config>{});
    };

    // This variable controls which layers will be compiled
    static constexpr auto ALL_LAYERS = std::tuple_cat(BigArch::layers);
    void
    register_all_layers(KernelRegistry&);
} // namespace chepp::nnue

#endif // CHEPP_LAYERS_H
