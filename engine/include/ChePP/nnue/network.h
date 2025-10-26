#ifndef CHEPP_NETWORK_H_
#define CHEPP_NETWORK_H_

#include "../engine/layers.h"
#include "affine.h"
#include "meta.h"
#include "relu.h"

#include <cstddef>
#include <hwy/highway.h>
#include <tuple>
#include <utility>

#include "affine-inl.h"

namespace chepp::nnue::network {
    using namespace meta;
    using namespace layers;

    template <typename... Layers>
    struct Network {
        std::tuple<std::shared_ptr<Layers>...> m_layers{};
        explicit Network(std::shared_ptr<Layers>... layers) : m_layers(std::move(layers)...) {
        }
    };

    inline auto
    get_network(const size_t bucket) {
        using act0_t        = relu::Layer<Types<int16_t, int8_t>, relu::Shift<8>>;
        static auto act0    = make_layer<act0_t>({
               .dims = {2048, 2048},
        });
        using l1_t          = affine::Layer<Types<int8_t, int32_t>>;
        static auto l1      = make_layer<l1_t>({
                 .dims    = Dims{2048, 16},
                 .weights = l1_weights[bucket],
                 .biases  = l1_biases[bucket],
        });
        using act1_t        = relu::Layer<Types<int32_t, int8_t>, relu::Shift<8>>;
        static auto act1    = make_layer<act1_t>({
               .dims = {16, 16},
        });
        using l2_t          = affine::Layer<Types<int8_t, int32_t>>;
        static auto l2      = make_layer<l2_t>({
                 .dims    = Dims{16, 32},
                 .weights = l2_weights[bucket],
                 .biases  = l2_biases[bucket],
        });
        using act2_t        = relu::Layer<Types<int32_t, int8_t>, relu::Shift<8>>;
        static auto act2    = make_layer<act2_t>({
               .dims = {32, 32},
        });
        using l3_t          = affine::Layer<Types<int8_t, int32_t>>;
        static auto l3      = make_layer<l3_t>({
                 .dims    = Dims{32, 1},
                 .weights = out_weights[bucket],
                 .biases  = out_biases[bucket],
        });
        static auto network = Network{act0, l1, act1, l2, act2, l3};
        return network;
    }
} // namespace chepp::nnue::network

#endif // CHEPP_NETWORK_INL_H
