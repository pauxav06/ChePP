#ifndef CHEPP_NETWORK_H_
#define CHEPP_NETWORK_H_

#include "../engine/layers.h"
#include "accumulator.h"
#include "affine.h"
#include "meta.h"
#include "relu.h"

#include <cstddef>

namespace chepp::nnue::network {
    using namespace meta;

    namespace accumulator {
        namespace layer = layers::accumulator;

        using namespace layers;
        using namespace layer;

        using types = Types<int16_t, int16_t>;

        using scalar_params = ParamComb_t<Scalar, types>;

        using UnrollOptions = std::tuple<Unroll<8>, Unroll<16>>;
        using simd_params   = ParamComb_t<Simd, types, UnrollOptions>;
        using params        = tuple_cat_types_t<scalar_params, simd_params>;
    } // namespace accumulator

    namespace psqt {
        namespace layer = layers::accumulator;

        using namespace layers;
        using namespace layer;

        using types         = Types<int32_t, int32_t>;
        using scalar_params = ParamComb_t<Scalar, types>;

        using UnrollOptions = std::tuple<Unroll<1>>;
        using simd_params   = ParamComb_t<Simd, types, UnrollOptions>;
        using params        = tuple_cat_types_t<scalar_params, simd_params>;

        inline constexpr Dims dims{16000, 8};
    } // namespace psqt

    namespace act0 {
        namespace layer = layers::relu;
        using namespace layers;
        using namespace layer;

        using types = Types<int16_t, int8_t>;
        using shift = std::tuple<std::integral_constant<size_t, 8>>;

        using scalar_params = ParamComb_t<Scalar, types, shift>;

        using simd_params = ParamComb_t<Scalar, types, shift>;
        using params      = tuple_cat_types_t<scalar_params, simd_params>;

        inline constexpr Dims dims{2048, 2048};
    } // namespace act0

    namespace l1 {
        namespace layer = layers::affine;
        using namespace layers;
        using namespace layer;

        using types = Types<int8_t, int32_t>;

        using scalar_params = ParamComb_t<Scalar, types>;

        using UnrollOptions = std::tuple<Unroll<1>, Unroll<4>, Unroll<8>, Unroll<16>>;
        using DotOps        = std::tuple<SumOfMulQuadAcc, SumOfMulPairAcc>;
        using simd_params   = ParamComb_t<SimdColMaj, types, UnrollOptions, DotOps>;
        using params        = tuple_cat_types_t<scalar_params, simd_params>;

        inline constexpr Dims dims{2048, 16};
    } // namespace l1

    namespace act1 {
        namespace layer = layers::relu;
        using namespace layers;
        using namespace layer;

        using types = Types<int32_t, int8_t>;
        using shift = std::tuple<std::integral_constant<size_t, 8>>;

        using scalar_params = ParamComb_t<Scalar, types, shift>;

        using simd_params = ParamComb_t<Scalar, types, shift>;
        using params      = tuple_cat_types_t<scalar_params, simd_params>;

        inline constexpr Dims dims{16, 16};
    } // namespace act1

    namespace l2 {
        namespace layer = layers::affine;
        using namespace layers;
        using namespace layer;

        using types = Types<int8_t, int32_t>;

        using scalar_params = ParamComb_t<Scalar, types>;

        using UnrollOptions = std::tuple<Unroll<1>, Unroll<4>>;
        using DotOps        = std::tuple<SumOfMulQuadAcc, SumOfMulPairAcc>;
        using simd_params   = ParamComb_t<SimdColMaj, types, UnrollOptions, DotOps>;
        using params        = tuple_cat_types_t<scalar_params, simd_params>;

        inline constexpr Dims dims{16, 32};
    } // namespace l2

    namespace act2 {
        namespace layer = layers::relu;
        using namespace layers;
        using namespace layer;

        using types = Types<int32_t, int8_t>;
        using shift = std::tuple<std::integral_constant<size_t, 8>>;

        using scalar_params = ParamComb_t<Scalar, types, shift>;

        using simd_params = ParamComb_t<Scalar, types, shift>;
        using params      = tuple_cat_types_t<scalar_params, simd_params>;

        inline constexpr Dims dims{32, 32};
    } // namespace act2

    namespace l3 {
        namespace layer = layers::affine;
        using namespace layers;
        using namespace layer;

        using types = Types<int8_t, int32_t>;

        using scalar_params = ParamComb_t<Scalar, types>;
        using params        = scalar_params;

        inline constexpr Dims dims{32, 1};
    } // namespace l3

    template <typename Accum, typename PSQT>
    struct Accumulator {
        using accum_type = Accum::value_type;
        using psqt_type  = PSQT::value_type;
    };

    template <typename... Modules>
    struct Inference {
        static_assert(sizeof...(Modules) > 0, "Network must have at least one module");

        static constexpr size_t n_modules = sizeof...(Modules);

        using input_module_t  = std::tuple_element_t<0, std::tuple<Modules...>>;
        using output_module_t = std::tuple_element_t<n_modules - 1, std::tuple<Modules...>>;

        using input_type                  = input_module_t::input_type;
        static constexpr auto input_size  = input_module_t::input_size;
        using input_view_t                = input_module_t::input_view_type;
        using output_type                 = output_module_t::input_type;
        static constexpr auto output_size = output_module_t::input_size;
        using output_view_t               = output_module_t::input_view_type;

        inline static std::tuple<Modules...> modules{};

        using test = std::span<int, 8>;

        // std::tuple<typename ModuleTraits<Modules::index>::buffer_type...> buffers{};

        template <typename T>
        using span_t = std::span<const T>;

        void load_weights(
            const std::tuple<std::pair<span_t<typename Modules::weight_type>, span_t<typename Modules::bias_type>>...>&
                wb) {
            for_each_in_tuple(modules,
                              [&]<typename extent_type>(auto& module, [[maybe_unused]] extent_type index_const) {
                                  constexpr std::size_t I = extent_type::value;
                                  const auto& [w, b]      = std::get<I>(wb);
                                  module.load_weights(w, b);
                              });
        }

        void load_weights(
            const std::pair<span_t<typename Modules::weight_type>, span_t<typename Modules::bias_type>>&... wb) {
            load_weights(std::tuple{wb...});
        }

        template <std::size_t I = 0>
        void forward() {
            thread_local std::tuple<hwy::AlignedVector<typename Modules::input_type>...> inputs{};

            auto& module = std::get<I>(modules);
            auto  out    = std::get<I>(buffers).span;

            if constexpr (I == 0) {
                module.forward(input, out);
            } else {
                auto prev = std::get<I - 1>(buffers).span;
                module.forward(prev, out);
            }

            if constexpr (I + 1 < sizeof...(Modules)) {
                forward<I + 1>(input);
            }
        }

        template <std::size_t Last = sizeof...(Modules) - 1>
        auto output() {
            std::get<Last>(buffers).span;
        }
    };

    template <typename Tuple>
    using MakeNetwork = unpack_tuple<Inference, Tuple>;

    template <typename Tuple>
    using MakeNetworks = map_tuple<MakeNetwork, Tuple>;

    template <typename Tuple>
    using MakeNetworks_t = MakeNetworks<Tuple>::type;

} // namespace chepp::nnue::network

#endif // CHEPP_NETWORK_INL_H
