#include "../../../../cmake-build-debug-coverage/_deps/catch2-src/single_include/catch2/catch.hpp"
#include "relu.h"

#include <any>
#include <cmath>
#include <experimental/mdarray>
#include <experimental/mdspan>
#include <hwy/base.h>

#if defined(CHEPP_RELU_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_RELU_INL_H_
#undef CHEPP_RELU_INL_H_
#else
#define CHEPP_RELU_INL_H_
#endif

#include "utils-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers::relu {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace nn = chepp::nnue::HWY_NAMESPACE;

        using namespace std::experimental;
        using namespace meta;

        template <typename Params>
        struct Layer;

        // scalar also serves as fallback in case i simd kernels are not available due to size constraits
        template <KernelConcept Kernel, TypesConcept Types, DimsConcept Dims, IntegralConstantConcept S>
        struct Layer<std::tuple<Kernel, Types, Dims, S>> {
            using extent_type = size_t;
            using input_type  = Types::in;
            using output_type = Types::out;

            CONSTEXPR_EXTENT(input_size, Dims::in)
            CONSTEXPR_EXTENT(output_size, Dims::out)

            static_assert(std::is_integral_v<input_type> && std::is_integral_v<output_type>);
            static_assert(Dims::in == Dims::out);

            static constexpr extent_type shift = S::value;

            static constexpr input_type max  = static_cast<input_type>(std::numeric_limits<output_type>::max());
            static constexpr input_type zero = static_cast<input_type>(0);

            static void load_weights([[maybe_unused]] std::any, [[maybe_unused]] std::any) {}

            static void forward(std::span<const input_type> input_view, std::span<output_type> output_view) {
                HWY_ASSERT(input_view.size() == input_size);
                HWY_ASSERT(output_view.size() == output_size);

                auto* HWY_RESTRICT input_ptr  = input_view.data();
                auto* HWY_RESTRICT output_ptr = output_view.data();

                std::span<const input_type, input_size> input{input_ptr, input_size};
                std::span<output_type, output_size>     output{output_ptr, output_size};

                for (extent_type i = 0; i < input.size(); ++i) {
                    input_type val = input[i];
                    if constexpr (shift != 0) val >>= shift;
                    val       = std::max(zero, val);
                    val       = std::min(max, val);
                    output[i] = static_cast<output_type>(val);
                }
            }
        };

        template <TypesConcept Types, DimsConcept Dims, IntegralConstantConcept S>
            requires(utils::is_power_of_two(Dims::in) &&
                     utils::is_power_of_two(sizeof(typename Types::in) / sizeof(typename Types::out)))
        struct Layer<std::tuple<Simd, Types, Dims, S>> {
            using extent_type = size_t;
            using input_type  = Types::in;
            using output_type = Types::out;

            CONSTEXPR_EXTENT(input_size, Dims::in)
            CONSTEXPR_EXTENT(output_size, Dims::out)

            static constexpr extent_type shift = S::value;
            CONSTEXPR_EXTENT(reductions, sizeof(input_type) / sizeof(output_type))

            static_assert(reductions > 1);
            static_assert(std::is_integral_v<input_type> && std::is_integral_v<output_type>);
            static_assert(input_size == output_size);

            using Din  = hn::ScalableTag<input_type>;
            using Vin  = hn::VFromD<Din>;
            using Dout = hn::ScalableTag<output_type>;
            using Vout = hn::VFromD<Dout>;

            MAYBE_CONSTEXPR_EXTENT(in_lanes, hn::Lanes(Din()));
            MAYBE_CONSTEXPR_EXTENT(out_lanes, hn::Lanes(Dout()));
            MAYBE_CONSTEXPR_EXTENT(padded_size, utils::pad_up(1 * input_size, 1 * in_lanes));
            MAYBE_CONSTEXPR_EXTENT(input_chunks, input_size / in_lanes);
            MAYBE_CONSTEXPR_EXTENT(output_chunks, input_size / out_lanes);

            static_assert(input_chunks / reductions == output_chunks);

            inline static HWY_LANES_CONSTEXPR auto s_input = utils::make_tensor<input_type>(input_chunks, input_size);
            inline static HWY_LANES_CONSTEXPR auto s_packed_input =
                utils::make_tensor<input_type>(output_chunks, reductions, in_lanes);
            inline static HWY_LANES_CONSTEXPR auto s_output = utils::make_tensor<output_type>(output_chunks, out_lanes);

            static void load_weights([[maybe_unused]] std::any, [[maybe_unused]] std::any) {}

            template <size_t N, typename Tag, typename GetRegFn>
            static auto ordered_demote_tree(GetRegFn&& get_reg) {
                if constexpr (N == 1) {
                    return get_reg(0);
                } else {
                    constexpr size_t half = N / 2;
                    auto             get  = [&](const size_t i) {
                        return hn::OrderedDemote2To(hn::RepartitionToNarrow<Tag>(), get_reg(2 * i), get_reg(2 * i + 1));
                    };
                    return ordered_demote_tree<half, hn::RepartitionToNarrow<Tag>>(get);
                }
            }

            static void forward(std::span<const input_type> input_view, std::span<output_type> output_view) {
                HWY_ASSERT(input_view.size() == input_size);
                HWY_ASSERT(output_view.size() == output_size);

                auto* HWY_RESTRICT input_ptr  = input_view.data();
                auto* HWY_RESTRICT output_ptr = output_view.data();
                const auto         in         = s_packed_input.make_const_span(input_ptr, s_packed_input.extent());
                const auto         out        = s_output.make_span(output_ptr, s_output.extent());

                for (extent_type c = 0; c < in.extent(0); c++) {
                    nn::RegisterBank<reductions, Vin>::run(
                        [&](const size_t i) { return hn::Load(Din(), &in[c, i, 0]); },
                        [&](auto get_reg, auto set_reg) {
                            if constexpr (shift != 0) {
                                for (int r = 0; r < reductions; r++) {
                                    set_reg(r, hn::ShiftRight<shift>(get_reg(r)));
                                }
                            }
                            Vout v_out = ordered_demote_tree<reductions, Din>(get_reg);
                            v_out      = hn::Max(hn::Zero(Dout()), v_out);
                            hn::Store(v_out, Dout(), &out[c, 0]);
                        });
                }
            }
        };
    }; // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::relu

HWY_AFTER_NAMESPACE();

#endif // CHEPP_RELU_INL_H
