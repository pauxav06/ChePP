#include <hwy/aligned_allocator.h>

#include <tuple>
#include <utility>

#include "affine.h"
#include "matrix.h"
#include "utils.h"

#include <memory>

#if defined(CHEPP_AFFINE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_AFFINE_INL_H_
#undef CHEPP_AFFINE_INL_H_
#else
#define CHEPP_AFFINE_INL_H_
#endif

#include "layer-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;

        using namespace hwy;
        using namespace utils;

        template <typename InT, size_t IS, typename OutT, size_t OS, typename WT, typename BT, default_config_t cfg>
            requires(std::is_same_v<default_config_t, decltype(cfg)>)
        struct Kernel<Affine<InT, IS, OutT, OS, WT, BT>, cfg> final : Affine<InT, IS, OutT, OS, WT, BT>::ikernel_t {
            using affine_t = Affine<InT, IS, OutT, OS, WT, BT>;
            using layer_t  = affine_t::layer_t;
            using base_t   = affine_t::ikernel_t;

            using base_t::base_t;
            using base_t::layer;

            using weight_t          = affine_t::weight_t;
            using bias_t            = affine_t::bias_t;
            using weights_extents_t = md::extents<size_t, affine_t::output_size_v, affine_t::input_size_v>;
            using biases_extents_t  = md::extents<size_t, affine_t::output_size_v>;

            using input_t  = affine_t::input_t;
            using output_t = affine_t::output_t;

            using input_extents_t  = md::extents<size_t, affine_t::input_size_v>;
            using output_extents_t = md::extents<size_t, affine_t::output_size_v>;

            using weights_t = md::mdspan<const weight_t, weights_extents_t>;
            using biases_t  = md::mdspan<const bias_t, biases_extents_t>;

            [[nodiscard]] std::string
            name() const noexcept override {
                return std::format("default,target={}", TargetName(HWY_TARGET));
            }

            explicit Kernel(const std::shared_ptr<layer_t>& l) noexcept
                : base_t(l), m_weights_data(std::begin(l->weights()), std::end(l->weights())),
                  m_biases_data(std::begin(l->biases()), std::end(l->biases())), m_weights(std::data(m_weights_data)),
                  m_biases(std::data(m_biases_data)) {
            }

            void
            forward(const input_t* HWY_RESTRICT input, output_t* HWY_RESTRICT output) const noexcept override {
                for (extent_type row = 0; row < m_weights.extent(0); ++row) {
                    output_t acc = 0;
                    for (extent_type col = 0; col < m_weights.extent(1); ++col) {
                        acc +=
                            static_cast<output_t>(MD_ACCESS(m_weights, row, col)) * static_cast<output_t>(input[col]);
                    }
                    output[row] = acc + MD_ACCESS(m_biases, row);
                }
            }

          private:
            hwy::AlignedVector<weight_t> m_weights_data;
            hwy::AlignedVector<bias_t>   m_biases_data;
            weights_t                    m_weights;
            biases_t                     m_biases;
        };

        template <typename InT, size_t IS, typename OutT, size_t OS, typename WT, typename BT, AffineSimdColMaj cfg>
            requires(std::is_same_v<AffineSimdColMaj, decltype(cfg)> &&
                     std::is_same_v<std::tuple<InT, OutT, WT, BT>, std::tuple<uint8_t, int32_t, int8_t, int32_t>> &&
                     is_power_of_two(cfg.unroll))
        struct Kernel<Affine<InT, IS, OutT, OS, WT, BT>, cfg> final : Affine<InT, IS, OutT, OS, WT, BT>::ikernel_t {
            using affine_t = Affine<InT, IS, OutT, OS, WT, BT>;
            using layer_t  = affine_t::layer_t;
            using base_t   = affine_t::ikernel_t;

            using base_t::base_t;
            using base_t::layer;

            using weight_t = affine_t::weight_t;
            using bias_t   = affine_t::bias_t;
            using input_t  = affine_t::input_t;
            using output_t = affine_t::output_t;

            static constexpr extent_type unroll{cfg.unroll};
            static constexpr extent_type pack{sizeof(output_t) / sizeof(input_t)};

            using Dw      = hn::ScalableTag<weight_t>;
            using Din     = hn::ScalableTag<input_t>;
            using Dpacked = hn::RepartitionToWideX2<Din>;
            using Dout    = hn::ScalableTag<output_t>;
            using Vw      = hn::VFromD<Dw>;
            using Vin     = hn::VFromD<Din>;
            using Vpacked = hn::VFromD<Dpacked>;
            using Vout    = hn::VFromD<Dout>;

            using packed_type = hn::TFromD<Dpacked>;

            HWY_STATIC_CONSTEXPR size_t m_input_lanes{hn::Lanes(Din())};
            HWY_STATIC_CONSTEXPR size_t m_output_lanes{hn::Lanes(Dout())};
            static constexpr size_t     m_padded_input_size{pad_up(affine_t::input_size_v, pack* unroll)};
            static constexpr size_t     m_input_padding{m_padded_input_size - affine_t::input_size_v};
            static constexpr size_t     m_chunks{m_padded_input_size / (pack * unroll)};
            HWY_STATIC_CONSTEXPR size_t m_padded_output_size{pad_up(affine_t::output_size_v, m_output_lanes)};
            HWY_STATIC_CONSTEXPR size_t m_output_padding{m_padded_output_size - affine_t::output_size_v};
            HWY_STATIC_CONSTEXPR size_t m_blocks{m_padded_output_size / m_output_lanes};

            using weights_extents_t = md::extents<size_t,
                                                  HWY_CONSTEXPR_EXT(m_blocks),
                                                  HWY_CONSTEXPR_EXT(m_chunks),
                                                  unroll,
                                                  HWY_CONSTEXPR_EXT(hn::Lanes(Dw()))>;
            using weights_t         = md::mdspan<const weight_t, weights_extents_t>;

            using biases_extents_t =
                md::extents<size_t, HWY_CONSTEXPR_EXT(m_blocks), HWY_CONSTEXPR_EXT(m_output_lanes)>;
            using biases_t = md::mdspan<const bias_t, biases_extents_t>;

            using input_view_extents_t  = md::extents<size_t, HWY_CONSTEXPR_EXT(m_chunks), unroll, pack>;
            using output_view_extents_t = biases_extents_t;

            static constexpr input_view_extents_t      m_input_view_extents{m_chunks, unroll, pack};
            HWY_STATIC_CONSTEXPR output_view_extents_t m_output_view_extents{m_blocks, m_output_lanes};

            [[nodiscard]] std::string
            name() const noexcept override {
                return std::format("simd-column-major,target={},unroll={},operation={}",
                                   hwy::TargetName(HWY_TARGET),
                                   unroll,
                                   (cfg.operation == AffineOperation::SumOfMulQuadAdd ? "VNNI" : "fallback"));
            }

            [[nodiscard]] size_t
            input_padding() const noexcept override {
                return m_input_padding;
            }

            [[nodiscard]] size_t
            output_padding() const noexcept override {
                return m_output_padding;
            }

            [[nodiscard]] static Vout
            dot(const Vin a, const Vw b, const Vout acc) noexcept {
                if constexpr (cfg.operation == AffineOperation::SumOfMulQuadAdd) {
                    return hn::SumOfMulQuadAccumulate(Dout(), a, b, acc);
                } else if constexpr (cfg.operation == AffineOperation::MulPairwiseAdd) {
                    using Dhalf      = hn::RepartitionToNarrow<Dout>;
                    using Vhalf      = hn::VFromD<Dhalf>;
                    const Vhalf sum0 = hn::SatWidenMulPairwiseAdd(Dhalf(), a, b);
                    const Vout  sum1 = hn::WidenMulPairwiseAdd(Dout(), sum0, hn::Set(Dhalf(), 1));
                    return hn::Add(acc, sum1);
                }
                std::unreachable();
            }

            explicit Kernel(const std::shared_ptr<layer_t>& l) noexcept
                : base_t(l), m_weights_data(m_padded_input_size * m_padded_output_size),
                  m_biases_data(m_padded_output_size),
                  m_weights(std::data(m_weights_data), m_blocks, m_chunks, unroll, hn::Lanes(Dw())),
                  m_biases(std::data(m_biases_data), m_blocks, m_output_lanes) {
                const matrix::MatrixView w{
                    std::data(layer().weights()), affine_t::output_size_v, affine_t::input_size_v};
                const auto w_t0 = pad(w, m_output_padding, m_input_padding);
                const auto w_t1 = tile_cols(w_t0, pack);
                const auto w_t2 = tile_cols(w_t1, m_input_lanes);

                const matrix::MatrixView b{std::data(layer().biases()), affine_t::output_size_v, 1};
                const auto               b_t0 = pad(b, m_output_padding, 0);

                materialize(w_t2, std::data(m_weights_data));
                materialize(b_t0, std::data(m_biases_data));
            }

            void
            forward(const input_t* HWY_RESTRICT in_ptr, output_t* HWY_RESTRICT out_ptr) const noexcept override {
                md::mdspan input{in_ptr, m_input_view_extents};
                md::mdspan output{out_ptr, m_output_view_extents};

                DECLARE_REG_BANK(weights_t::static_extent(2), Vout);
                for (extent_type b = 0; b < m_weights.extent(0); ++b) {
                    for (size_t u = 0; u < m_weights.extent(2); ++u) {
                        *regs[u] = hn::Zero(Dout());
                    }
                    *regs[0] = hn::Load(Dout(), &MD_ACCESS(m_biases, b, 0));
                    for (extent_type c = 0; c < m_weights.extent(1); ++c) {
                        for (size_t u = 0; u < m_weights.extent(2); ++u) {
                            packed_type packed{0};
                            std::memcpy(&packed, &MD_ACCESS(input, c, u, 0), sizeof(packed_type)); // cast is UB
                            auto in  = hn::Set(Dpacked(), packed);
                            *regs[u] = dot(
                                hn::BitCast(Din(), in), hn::Load(Dw(), &MD_ACCESS(m_weights, b, c, u, 0)), *regs[u]);
                        }
                    }
                    constexpr_for<0, std::countr_zero(weights_t::static_extent(2))>([&](auto i) {
                        constexpr auto off = 1uz << i;
                        constexpr_for<0, weights_t::static_extent(2), 2 * off>(
                            [&](auto j) { *regs[j] += *regs[j + off]; });
                    });
                    hn::Store(*regs[0], Dout(), &MD_ACCESS(output, b, 0));
                }
            }

          private:
            hwy::AlignedVector<weight_t> m_weights_data;
            hwy::AlignedVector<output_t> m_biases_data;
            weights_t                    m_weights;
            biases_t                     m_biases;
        };

        template <typename InT, size_t IS, typename OutT, size_t OS, typename WT, typename BT, AffineSimdRowMaj cfg>
            requires(std::is_same_v<AffineSimdRowMaj, decltype(cfg)> &&
                     std::is_same_v<std::tuple<InT, OutT, WT, BT>, std::tuple<uint8_t, int32_t, int8_t, int32_t>> &&
                     is_power_of_two(cfg.unroll))
        struct Kernel<Affine<InT, IS, OutT, OS, WT, BT>, cfg> final : Affine<InT, IS, OutT, OS, WT, BT>::ikernel_t {
            using affine_t = Affine<InT, IS, OutT, OS, WT, BT>;
            using layer_t  = affine_t::layer_t;
            using base_t   = affine_t::ikernel_t;

            using base_t::base_t;
            using base_t::layer;

            using weight_t = affine_t::weight_t;
            using bias_t   = affine_t::bias_t;
            using input_t  = affine_t::input_t;
            using output_t = affine_t::output_t;

            static constexpr extent_type unroll{cfg.unroll};

            using Dw      = hn::ScalableTag<weight_t>;
            using Din     = hn::ScalableTag<input_t>;
            using Dpacked = hn::RepartitionToWideX2<Din>;
            using Dout    = hn::ScalableTag<output_t>;
            using Vw      = hn::VFromD<Dw>;
            using Vin     = hn::VFromD<Din>;
            using Vpacked = hn::VFromD<Dpacked>;
            using Vout    = hn::VFromD<Dout>;

            using packed_type = hn::TFromD<Dpacked>;

            HWY_STATIC_CONSTEXPR size_t m_input_lanes{hn::Lanes(Din())};
            HWY_STATIC_CONSTEXPR size_t m_output_lanes{hn::Lanes(Dout())};
            HWY_STATIC_CONSTEXPR size_t m_padded_input_size{pad_up(affine_t::input_size_v, m_input_lanes)};
            HWY_STATIC_CONSTEXPR size_t m_input_padding{m_padded_input_size - affine_t::input_size_v};
            HWY_STATIC_CONSTEXPR size_t m_chunks{m_padded_input_size / m_input_lanes};
            static constexpr size_t     m_padded_output_size{pad_up(affine_t::output_size_v, unroll)};
            static constexpr size_t     m_output_padding{m_padded_output_size - affine_t::output_size_v};
            static constexpr size_t     m_blocks{m_padded_output_size / unroll};

            using weights_extents_t =
                md::extents<size_t, m_blocks, HWY_CONSTEXPR_EXT(m_chunks), unroll, HWY_CONSTEXPR_EXT(hn::Lanes(Dw()))>;
            using weights_t = md::mdspan<const weight_t, weights_extents_t>;

            using biases_extents_t = md::extents<size_t, m_blocks, unroll>;
            using biases_t         = md::mdspan<const bias_t, biases_extents_t>;

            using input_view_extents_t =
                md::extents<size_t, HWY_CONSTEXPR_EXT(m_chunks), HWY_CONSTEXPR_EXT(m_input_lanes)>;
            using output_view_extents_t = biases_extents_t;

            HWY_STATIC_CONSTEXPR input_view_extents_t m_input_view_extents{m_chunks, m_input_lanes};
            static constexpr output_view_extents_t    m_output_view_extents{m_blocks, unroll};

            [[nodiscard]] std::string
            name() const noexcept override {
                return std::format("simd-row-major,target={},unroll={},operation={}",
                                   hwy::TargetName(HWY_TARGET),
                                   unroll,
                                   (cfg.operation == AffineOperation::SumOfMulQuadAdd ? "VNNI" : "fallback"));
            }

            [[nodiscard]] size_t
            input_padding() const noexcept override {
                return m_input_padding;
            }

            [[nodiscard]] size_t
            output_padding() const noexcept override {
                return m_output_padding;
            }

            [[nodiscard]] static Vout
            dot(const Vin a, const Vw b, const Vout acc) noexcept {
                if constexpr (cfg.operation == AffineOperation::SumOfMulQuadAdd) {
                    return hn::SumOfMulQuadAccumulate(Dout(), a, b, acc);
                } else if constexpr (cfg.operation == AffineOperation::MulPairwiseAdd) {
                    using Dhalf      = hn::RepartitionToNarrow<Dout>;
                    using Vhalf      = hn::VFromD<Dhalf>;
                    const Vhalf sum0 = hn::SatWidenMulPairwiseAdd(Dhalf(), a, b);
                    const Vout  sum1 = hn::WidenMulPairwiseAdd(Dout(), sum0, hn::Set(Dhalf(), 1));
                    return hn::Add(acc, sum1);
                }
                std::unreachable();
            }

            explicit Kernel(const std::shared_ptr<layer_t>& l) noexcept
                : base_t(l), m_weights_data(m_padded_input_size * m_padded_output_size),
                  m_biases_data(m_padded_output_size),

                  m_weights(std::data(m_weights_data), m_blocks, m_chunks, unroll, hn::Lanes(Dw())),
                  m_biases(std::data(m_biases_data), m_blocks, unroll) {
                const matrix::MatrixView w{
                    std::data(layer().weights()), affine_t::output_size_v, affine_t::input_size_v};
                const auto w_t0 = pad(w, m_output_padding, m_input_padding);
                const auto w_t1 = tile_cols(w_t0, m_input_lanes);
                const auto w_t2 = hsplit(w_t1, m_blocks);

                const matrix::MatrixView b{std::data(layer().biases()), affine_t::output_size_v, 1};
                const auto               b_t0 = pad(b, m_output_padding, 0);

                materialize(w_t2, std::data(m_weights_data));
                materialize(b_t0, std::data(m_biases_data));
            }

            void
            forward(const input_t* HWY_RESTRICT in_ptr, output_t* HWY_RESTRICT out_ptr) const noexcept override {
                md::mdspan input{in_ptr, m_input_view_extents};
                md::mdspan output{out_ptr, m_output_view_extents};

                DECLARE_REG_BANK(weights_t::static_extent(2), Vout);
                for (extent_type b = 0; b < m_weights.extent(0); ++b) {
                    for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                        GET_REG(u) = hn::Zero(Dout());
                    }
                    for (extent_type c = 0; c < m_weights.extent(1); ++c) {
                        auto in = hn::Load(Din(), &MD_ACCESS(input, c, 0));
                        for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                            GET_REG(u) = dot(
                                hn::BitCast(Din(), in), hn::Load(Dw(), &MD_ACCESS(m_weights, b, c, u, 0)), GET_REG(u));
                        }
                    }
                    for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                        MD_ACCESS(output, b, u) = hn::ReduceSum(Dout(), GET_REG(u)) + MD_ACCESS(m_biases, b, u);
                    }
                }
            }

          private:
            hwy::AlignedVector<weight_t> m_weights_data;
            hwy::AlignedVector<bias_t>   m_biases_data;
            weights_t                    m_weights;
            biases_t                     m_biases;
        };
    }; // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers
HWY_AFTER_NAMESPACE();

#endif
