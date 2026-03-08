#include "mdspan.h"

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

        template <typename InT, size_t IS, typename OutT, unsigned Q, default_config_t cfg>
            requires(std::is_same_v<default_config_t, decltype(cfg)>)
        struct Kernel<ClippedRelu<InT, IS, OutT, Q>, cfg> final : ClippedRelu<InT, IS, OutT, Q>::ikernel_t {
            using relu_t  = ClippedRelu<InT, IS, OutT, Q>;
            using base_t  = relu_t::ikernel_t;
            using layer_t = relu_t::layer_t;

            using base_t::base_t;
            using base_t::layer;

            using input_t  = relu_t::input_t;
            using output_t = relu_t::output_t;

            using input_extents_t  = md::extents<size_t, relu_t::size_v>;
            using output_extents_t = md::extents<size_t, relu_t::size_v>;

            [[nodiscard]] std::string
            name() const noexcept override {
                return fmt::format("default,target={}", hwy::TargetName(HWY_TARGET));
            }

            void
            forward(const input_t* HWY_RESTRICT input_ptr, output_t* HWY_RESTRICT output_ptr) const noexcept override {
                md::mdspan input{input_ptr, input_extents_t{}};
                md::mdspan output{output_ptr, output_extents_t{}};
                for (size_t i = 0; i < input.extent(0); ++i) {
                    input_t val = MD_ACCESS(input, i);
                    if constexpr (relu_t::shift != 0) {
                        val /= relu_t::quantize;
                    }
                    val                  = std::max(relu_t::min, val);
                    val                  = std::min(relu_t::max, val);
                    MD_ACCESS(output, i) = static_cast<output_t>(val);
                }
            }
        };

        template <typename InT, size_t IS, typename OutT, unsigned Q, ClippedReluSimd cfg>
            requires(std::is_same_v<ClippedReluSimd, decltype(cfg)> && cfg.unroll * (sizeof(InT) / sizeof(OutT)) <= 32)
        struct Kernel<ClippedRelu<InT, IS, OutT, Q>, cfg> final : ClippedRelu<InT, IS, OutT, Q>::ikernel_t {
            using relu_t  = ClippedRelu<InT, IS, OutT, Q>;
            using base_t  = relu_t::ikernel_t;
            using layer_t = relu_t::layer_t;

            using base_t::base_t;
            using base_t::layer;

            using input_t  = relu_t::input_t;
            using output_t = relu_t::output_t;

            static constexpr extent_type factor{sizeof(input_t) / sizeof(output_t)};
            static constexpr extent_type unroll{cfg.unroll};

            using Din   = hn::ScalableTag<input_t>;
            using Vin   = hn::VFromD<Din>;
            using Dout  = hn::ScalableTag<output_t>;
            using Vout  = hn::VFromD<Dout>;
            using Douts = hn::ScalableTag<std::make_signed_t<output_t>>;
            using Vouts = hn::VFromD<Douts>;

            HWY_STATIC_CONSTEXPR size_t m_input_lanes{hn::Lanes(Din())};
            HWY_STATIC_CONSTEXPR size_t m_output_lanes{hn::Lanes(Dout())};
            HWY_STATIC_CONSTEXPR size_t m_padded_input_size{pad_up(relu_t::size_v, unroll* m_input_lanes* factor)};
            HWY_STATIC_CONSTEXPR size_t m_input_padding{m_padded_input_size - relu_t::size_v};
            HWY_STATIC_CONSTEXPR size_t m_padded_output_size{pad_up(relu_t::size_v, unroll* m_output_lanes)};
            HWY_STATIC_CONSTEXPR size_t m_output_padding{m_padded_output_size - relu_t::size_v};
            HWY_STATIC_CONSTEXPR size_t m_input_chunks{m_padded_input_size / (m_input_lanes * unroll * factor)};
            HWY_STATIC_CONSTEXPR size_t m_output_chunks{m_padded_output_size / (m_output_lanes * unroll)};

            using input_extents_t = md::
                extents<size_t, HWY_CONSTEXPR_EXT(m_input_chunks), unroll, factor, HWY_CONSTEXPR_EXT(m_input_lanes)>;
            using output_extents_t =
                md::extents<size_t, HWY_CONSTEXPR_EXT(m_output_chunks), unroll, HWY_CONSTEXPR_EXT(m_output_lanes)>;
            using input_viewt_t  = md::mdspan<const input_t, input_extents_t>;
            using output_viewt_t = md::mdspan<output_t, output_extents_t>;

            HWY_STATIC_CONSTEXPR input_extents_t  m_input_extents{m_input_chunks, unroll, factor, m_input_lanes};
            HWY_STATIC_CONSTEXPR output_extents_t m_output_extents{m_output_chunks, unroll, m_output_lanes};

            [[nodiscard]] std::string
            name() const noexcept override {
                return fmt::format("Simd,target={},unroll={}", hwy::TargetName(HWY_TARGET), unroll);
            }

            [[nodiscard]] size_t
            input_padding() const noexcept override {
                return m_input_padding;
            }
            [[nodiscard]] size_t
            output_padding() const noexcept override {
                return m_output_padding;
            }

            template <size_t N, typename V>
            [[nodiscard]] HWY_INLINE static auto
            ordered_demote_tree(std::span<V*, N> in_regs) noexcept {
                if constexpr (N == 1) {
                    return *in_regs[0];
                } else {
                    using D               = hn::DFromV<V>;
                    using Dhalf           = hn::RepartitionToNarrow<D>;
                    using Vhalf           = hn::VFromD<Dhalf>;
                    constexpr size_t half = N / 2;
                    DECLARE_REG_BANK(half, Vhalf)
                    constexpr_for<0, half>([&](const size_t i) {
                        *regs[i] = hn::OrderedDemote2To(Dhalf(), *in_regs[2 * i], *in_regs[2 * i + 1]);
                    });
                    return ordered_demote_tree<half>(regs);
                }
            }

            void
            forward(const input_t* HWY_RESTRICT input_ptr, output_t* HWY_RESTRICT output_ptr) const noexcept override {
                input_viewt_t  input{input_ptr, m_input_extents};
                output_viewt_t output{output_ptr, m_output_extents};

                DECLARE_REG_BANK(unroll * factor, Vin)
                for (extent_type c = 0; c < input.extent(0); ++c) {
                    for (size_t u = 0; u < input.extent(1); ++u) {
                        for (size_t v = 0; v < input.extent(2); ++v) {
                            auto idx     = u * input.extent(2) + v;
                            GET_REG(idx) = hn::Load(Din(), &MD_ACCESS(input, c, u, v, 0));
                            if constexpr (relu_t::shift != 0) {
                                GET_REG(idx) = hn::ShiftRight<relu_t::shift>(GET_REG(idx));
                            }
                        }
                    }
                    constexpr_for<0, unroll>([&](auto U) {
                        Vouts v_out_s = ordered_demote_tree<factor>(regs.template subspan<U * factor, factor>());
                        Vout  v_out   = hn::BitCast(Dout(), hn::Max(hn::Zero(Douts()), v_out_s));
                        hn::Store(v_out, Dout(), &MD_ACCESS(output, c, U, 0));
                    });
                }
            }
        };
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers

HWY_AFTER_NAMESPACE();

#endif // CHEPP_RELU_INL_H
