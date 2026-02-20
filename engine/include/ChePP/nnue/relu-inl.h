#include <mdspan>

#include "relu.h"
#include "utils.h"

#if defined(CHEPP_RELU_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_RELU_INL_H_
#undef CHEPP_RELU_INL_H_
#else
#define CHEPP_RELU_INL_H_
#endif

#include "layer-inl.h"

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        using namespace utils;

        template <typename InT, std::size_t IS, typename OutT, unsigned Q, default_config_t cfg>
        struct Kernel<ClippedReLULayer<InT, IS, OutT, Q>, cfg> final : ClippedReLULayer<InT, IS, OutT, Q>::IKernel {
            using Layer            = ClippedReLULayer<InT, IS, OutT, Q>;
            using extent_type      = Layer::extent_type;
            using input_type       = Layer::input_type;
            using output_type      = Layer::output_type;
            using input_extents_t  = std::extents<std::size_t, Layer::size()>;
            using output_extents_t = std::extents<std::size_t, Layer::size()>;

            explicit Kernel(std::shared_ptr<const Layer> layer) : m_layer(layer) {
            }

            void
            forward(const input_type* HWY_RESTRICT input_ptr, output_type* HWY_RESTRICT output_ptr) const override {
                std::mdspan input{input_ptr, input_extents_t{}};
                std::mdspan output{output_ptr, output_extents_t{}};
                for (std::size_t i = 0; i < input.extent(0); ++i) {
                    input_type val = input[i];
                    if constexpr (Layer::shift != 0) {
                        val /= Layer::quantize;
                    }
                    val       = std::max(Layer::min, val);
                    val       = std::min(Layer::max, val);
                    output[i] = static_cast<output_type>(val);
                }
            }

          private:
            std::shared_ptr<const Layer> m_layer;
        };

        template <typename InT, std::size_t IS, typename OutT, unsigned Q, ClippedReluSimd cfg>
            requires(cfg.unroll * (sizeof(InT) / sizeof(OutT)) <= 32)
        struct Kernel<ClippedReLULayer<InT, IS, OutT, Q>, cfg> final : ClippedReLULayer<InT, IS, OutT, Q>::IKernel {
            using Layer       = ClippedReLULayer<InT, IS, OutT, Q>;
            using extent_type = Layer::extent_type;
            using input_type  = Layer::input_type;
            using output_type = Layer::output_type;

            static constexpr extent_type factor{sizeof(input_type) / sizeof(output_type)};
            static constexpr extent_type unroll{cfg.unroll};

            using Din   = hn::ScalableTag<input_type>;
            using Vin   = hn::VFromD<Din>;
            using Dout  = hn::ScalableTag<output_type>;
            using Vout  = hn::VFromD<Dout>;
            using Douts = hn::ScalableTag<std::make_signed_t<output_type>>;
            using Vouts = hn::VFromD<Douts>;

            HWY_STATIC_CONSTEXPR std::size_t m_input_lanes{hn::Lanes(Din())};
            HWY_STATIC_CONSTEXPR std::size_t m_output_lanes{hn::Lanes(Dout())};
            HWY_STATIC_CONSTEXPR std::size_t m_padded_input_size{pad_up(Layer::size(), unroll* m_input_lanes* factor)};
            HWY_STATIC_CONSTEXPR std::size_t m_input_padding{m_padded_input_size - Layer::size()};
            HWY_STATIC_CONSTEXPR std::size_t m_padded_output_size{pad_up(Layer::size(), unroll* m_output_lanes)};
            HWY_STATIC_CONSTEXPR std::size_t m_output_padding{m_padded_output_size - Layer::size()};
            HWY_STATIC_CONSTEXPR std::size_t m_input_chunks{m_padded_input_size / (m_input_lanes * unroll * factor)};
            HWY_STATIC_CONSTEXPR std::size_t m_output_chunks{m_padded_output_size / (m_output_lanes * unroll)};

            using input_extents_t  = std::extents<std::size_t,
                                                 HWY_CONSTEXPR_EXT(m_input_chunks),
                                                 unroll,
                                                 factor,
                                                 HWY_CONSTEXPR_EXT(m_input_lanes)>;
            using output_extents_t = std::
                extents<std::size_t, HWY_CONSTEXPR_EXT(m_output_chunks), unroll, HWY_CONSTEXPR_EXT(m_output_lanes)>;
            using input_viewt_t  = std::mdspan<const input_type, input_extents_t>;
            using output_viewt_t = std::mdspan<output_type, output_extents_t>;

            HWY_STATIC_CONSTEXPR input_extents_t  m_input_extents{m_input_chunks, unroll, factor, m_input_lanes};
            HWY_STATIC_CONSTEXPR output_extents_t m_output_extents{m_output_chunks, unroll, m_output_lanes};

            explicit Kernel(std::shared_ptr<const Layer> layer) : m_layer(layer) {
            }

            [[nodiscard]] std::size_t
            input_padding() const override {
                return m_input_padding;
            }
            [[nodiscard]] std::size_t
            output_padding() const override {
                return m_output_padding;
            }

            void
            forward(const input_type* HWY_RESTRICT input_ptr, output_type* HWY_RESTRICT output_ptr) const override {
                input_viewt_t  input{input_ptr, m_input_extents};
                output_viewt_t output{output_ptr, m_output_extents};

                DECLARE_REG_BANK(unroll * factor, Vin)
                for (extent_type c = 0; c < input.extent(0); ++c) {
                    for (std::size_t u = 0; u < input.extent(1); ++u) {
                        for (std::size_t v = 0; v < input.extent(2); ++v) {
                            auto idx     = u * input.extent(2) + v;
                            GET_REG(idx) = hn::Load(Din(), &input[c, u, v, 0]);
                            if constexpr (Layer::shift != 0) {
                                GET_REG(idx) = hn::ShiftRight<Layer::shift>(GET_REG(idx));
                            }
                        }
                    }
                    constexpr_for<0, unroll>([&](auto U) {
                        Vouts v_out_s = ordered_demote_tree<factor>(regs.template subspan<U * factor, factor>());
                        Vout  v_out   = hn::BitCast(Dout(), hn::Max(hn::Zero(Douts()), v_out_s));
                        hn::Store(v_out, Dout(), &output[c, U, 0]);
                    });
                }
            }

          private:
            template <size_t N, typename V>
            [[nodiscard]] HWY_INLINE static auto
            ordered_demote_tree(std::span<V*, N> in_regs) {
                if constexpr (N == 1) {
                    return *in_regs[0];
                } else {
                    using D               = hn::DFromV<V>;
                    using Dhalf           = hn::RepartitionToNarrow<D>;
                    using Vhalf           = hn::VFromD<Dhalf>;
                    constexpr size_t half = N / 2;
                    DECLARE_REG_BANK(half, Vhalf)
                    constexpr_for<0, N>([&](const size_t i) {
                        GET_REG(i) = hn::OrderedDemote2To(Dhalf(), *in_regs[2 * i], *in_regs[2 * i + 1]);
                    });
                    return ordered_demote_tree<half>(regs);
                }
            }

            std::shared_ptr<const Layer> m_layer;
        };

    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers

HWY_AFTER_NAMESPACE();

#endif // CHEPP_RELU_INL_H
