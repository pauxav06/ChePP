#include <hwy/aligned_allocator.h>

#include <mdspan>
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

        template <typename InT, std::size_t IS, typename OutT, std::size_t OS, typename WT, typename BT, auto cfg>
        struct Kernel<AffineLayer<InT, IS, OutT, OS, WT, BT>, cfg> final
            : AffineLayer<InT, IS, OutT, OS, WT, BT>::IKernel {
            using Layer = AffineLayer<InT, IS, OutT, OS, WT, BT>;

            using weights_type      = Layer::weights_type;
            using biases_type       = Layer::biases_type;
            using weights_extents_t = std::extents<std::size_t, Layer::output_size(), Layer::input_size()>;
            using biases_extents_t  = std::extents<std::size_t, Layer::output_size()>;

            using input_type       = Layer::input_type;
            using output_type      = Layer::output_type;
            using input_extents_t  = std::extents<std::size_t, Layer::input_size()>;
            using output_extents_t = std::extents<std::size_t, Layer::output_size()>;

            explicit Kernel(const std::shared_ptr<const Layer>& layer)
                : m_layer(layer), m_weights{layer->weights().data(), weights_extents_t{}},
                  m_biases(layer->biases().data(), biases_extents_t{}) {
            }

            void
            forward(const input_type* HWY_RESTRICT input, output_type* HWY_RESTRICT output) const override {
                for (extent_type row = 0; row < m_weights.extent(0); ++row) {
                    output_type acc = 0;
                    for (extent_type col = 0; col < m_weights.extent(1); ++col) {
                        acc += static_cast<output_type>(m_weights[row, col]) * static_cast<output_type>(input[col]);
                    }
                    output[row] = acc + m_biases[row];
                }
            }

          private:
            std::shared_ptr<const Layer>                       m_layer;
            std::mdspan<const weights_type, weights_extents_t> m_weights;
            std::mdspan<const biases_type, biases_extents_t>   m_biases;
        };

        template <typename InT,
                  std::size_t IS,
                  typename OutT,
                  std::size_t OS,
                  typename WT,
                  typename BT,
                  AffineSimdColMaj cfg>
            requires(std::is_same_v<std::tuple<InT, OutT, WT, BT>, std::tuple<uint8_t, int32_t, int8_t, int32_t>> &&
                     is_power_of_two(cfg.unroll))
        struct Kernel<AffineLayer<InT, IS, OutT, OS, WT, BT>, cfg> final
            : AffineLayer<InT, IS, OutT, OS, WT, BT>::IKernel {
            using Layer = AffineLayer<InT, IS, OutT, OS, WT, BT>;

            using weights_type = Layer::weights_type;
            using biases_type  = Layer::biases_type;

            using input_type  = Layer::input_type;
            using output_type = Layer::output_type;

            static constexpr extent_type unroll{cfg.unroll};
            static constexpr extent_type pack{sizeof(output_type) / sizeof(input_type)};

            using Dw      = hn::ScalableTag<weights_type>;
            using Din     = hn::ScalableTag<input_type>;
            using Dpacked = hn::RepartitionToWideX2<Din>;
            using Dout    = hn::ScalableTag<output_type>;
            using Vw      = hn::VFromD<Dw>;
            using Vin     = hn::VFromD<Din>;
            using Vpacked = hn::VFromD<Dpacked>;
            using Vout    = hn::VFromD<Dout>;

            using packed_type = hn::TFromD<Dpacked>;

            HWY_STATIC_CONSTEXPR std::size_t m_input_lanes{hn::Lanes(Din())};
            HWY_STATIC_CONSTEXPR std::size_t m_output_lanes{hn::Lanes(Dout())};
            static constexpr std::size_t     m_padded_input_size{utils::pad_up(Layer::input_size(), pack* unroll)};
            static constexpr std::size_t     m_input_padding{m_padded_input_size - Layer::input_size()};
            static constexpr std::size_t     m_chunks{m_padded_input_size / (pack * unroll)};
            HWY_STATIC_CONSTEXPR std::size_t m_padded_output_size{utils::pad_up(Layer::output_size(), m_output_lanes)};
            HWY_STATIC_CONSTEXPR std::size_t m_output_padding{m_padded_output_size - Layer::output_size()};
            HWY_STATIC_CONSTEXPR std::size_t m_blocks{m_padded_output_size / m_output_lanes};

            using weights_extents_t = std::extents<std::size_t,
                                                   HWY_CONSTEXPR_EXT(m_blocks),
                                                   HWY_CONSTEXPR_EXT(m_chunks),
                                                   unroll,
                                                   HWY_CONSTEXPR_EXT(hn::Lanes(Dw()))>;
            using weights_view_t    = std::mdspan<const weights_type, weights_extents_t>;

            using biases_extents_t =
                std::extents<std::size_t, HWY_CONSTEXPR_EXT(m_blocks), HWY_CONSTEXPR_EXT(m_output_lanes)>;
            using biases_view_t = std::mdspan<const biases_type, biases_extents_t>;

            using input_view_extents_t  = std::extents<size_t, HWY_CONSTEXPR_EXT(m_chunks), unroll, pack>;
            using output_view_extents_t = biases_extents_t;

            static constexpr input_view_extents_t      m_input_view_extents{m_chunks, unroll, pack};
            HWY_STATIC_CONSTEXPR output_view_extents_t m_output_view_extents{m_blocks, m_output_lanes};

            explicit Kernel(const std::shared_ptr<const Layer>& layer)
                : m_layer(layer), m_weights_storage(m_padded_input_size * m_padded_output_size),
                  m_weights(m_weights_storage.data(), m_blocks, m_chunks, unroll, hn::Lanes(Dw())),
                  m_biases_storage(m_padded_output_size), m_biases(m_biases_storage.data(), m_blocks, m_output_lanes) {
                const matrix::MatrixView w{m_layer->weights().data(), m_layer->output_size(), m_layer->input_size()};
                const auto               w_t0 = pad(w, m_output_padding, m_input_padding);
                const auto               w_t1 = tile_cols(w_t0, pack);
                const auto               w_t2 = tile_cols(w_t1, m_input_lanes);

                const matrix::MatrixView b{m_layer->biases().data(), m_layer->output_size(), 1};
                const auto               b_t0 = pad(b, m_output_padding, 0);

                materialize(w_t2, m_weights_storage.data());
                materialize(b_t0, m_biases_storage.data());
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
            forward(const input_type* HWY_RESTRICT in_ptr, output_type* HWY_RESTRICT out_ptr) const override {
                std::mdspan input{in_ptr, m_input_view_extents};
                std::mdspan output{out_ptr, m_output_view_extents};

                DECLARE_REG_BANK(32, Vout);
                for (extent_type b = 0; b < m_weights.extent(0); ++b) {
                    for (std::size_t u = 0; u < m_weights.extent(2); ++u) {
                        GET_REG(u) = hn::Zero(Dout());
                    }
                    GET_REG(0) = hn::Load(Dout(), &m_biases[b, 0]);
                    for (extent_type c = 0; c < m_weights.extent(1); ++c) {
                        for (std::size_t u = 0; u < m_weights.extent(2); ++u) {
                            packed_type packed{0};
                            std::memcpy(&packed, &input[c, u, 0], sizeof(packed_type)); // cast is UB
                            auto in = hn::Set(Dpacked(), packed);
                            GET_REG(u) =
                                dot(hn::BitCast(Din(), in), hn::Load(Dw(), &m_weights[b, c, u, 0]), GET_REG(u));
                        }
                    }
                    utils::constexpr_for<0, std::countr_zero(m_weights.extent(2))>([&](auto i) {
                        constexpr auto off = 1uz << i;
                        constexpr_for<0, m_weights.extent(2), 2 * off>([&](auto j) { GET_REG(j) += GET_REG(j + off); });
                    });
                    hn::Store(GET_REG(0), Dout(), &output[b, 0]);
                }
            }

          private:
            [[nodiscard]] static Vout
            dot(const Vin a, const Vw b, const Vout acc) {
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

            std::shared_ptr<const Layer> m_layer;
            AlignedVector<weights_type>  m_weights_storage;
            weights_view_t               m_weights;
            AlignedVector<biases_type>   m_biases_storage;
            biases_view_t                m_biases;
        };

        template <typename InT,
                  std::size_t IS,
                  typename OutT,
                  std::size_t OS,
                  typename WT,
                  typename BT,
                  AffineSimdRowMaj cfg>
            requires(std::is_same_v<std::tuple<InT, OutT, WT, BT>, std::tuple<uint8_t, int32_t, int8_t, int32_t>> &&
                     is_power_of_two(cfg.unroll))
        struct Kernel<AffineLayer<InT, IS, OutT, OS, WT, BT>, cfg> : AffineLayer<InT, IS, OutT, OS, WT, BT>::IKernel {
            using Layer = AffineLayer<InT, IS, OutT, OS, WT, BT>;

            using weights_type = Layer::weights_type;
            using biases_type  = Layer::biases_type;

            using input_type  = Layer::input_type;
            using output_type = Layer::output_type;

            static constexpr extent_type unroll{cfg.unroll};

            using Dw      = hn::ScalableTag<weights_type>;
            using Din     = hn::ScalableTag<input_type>;
            using Dpacked = hn::RepartitionToWideX2<Din>;
            using Dout    = hn::ScalableTag<output_type>;
            using Vw      = hn::VFromD<Dw>;
            using Vin     = hn::VFromD<Din>;
            using Vpacked = hn::VFromD<Dpacked>;
            using Vout    = hn::VFromD<Dout>;

            using packed_type = hn::TFromD<Dpacked>;

            HWY_STATIC_CONSTEXPR std::size_t m_input_lanes{hn::Lanes(Din())};
            HWY_STATIC_CONSTEXPR std::size_t m_output_lanes{hn::Lanes(Dout())};
            HWY_STATIC_CONSTEXPR std::size_t m_padded_input_size{utils::pad_up(Layer::input_size(), m_input_lanes)};
            HWY_STATIC_CONSTEXPR std::size_t m_input_padding{m_padded_input_size - Layer::input_size()};
            HWY_STATIC_CONSTEXPR std::size_t m_chunks{m_padded_input_size / m_input_lanes};
            static constexpr std::size_t     m_padded_output_size{utils::pad_up(Layer::output_size(), unroll)};
            static constexpr std::size_t     m_output_padding{m_padded_output_size - Layer::output_size()};
            static constexpr std::size_t     m_blocks{m_padded_output_size / unroll};

            using weights_extents_t = std::
                extents<std::size_t, m_blocks, HWY_CONSTEXPR_EXT(m_chunks), unroll, HWY_CONSTEXPR_EXT(hn::Lanes(Dw()))>;
            using weights_view_t = std::mdspan<const weights_type, weights_extents_t>;

            using biases_extents_t = std::extents<std::size_t, m_blocks, unroll>;
            using biases_view_t    = std::mdspan<const biases_type, biases_extents_t>;

            using input_view_extents_t =
                std::extents<size_t, HWY_CONSTEXPR_EXT(m_chunks), HWY_CONSTEXPR_EXT(m_input_lanes)>;
            using output_view_extents_t = biases_extents_t;

            HWY_STATIC_CONSTEXPR input_view_extents_t m_input_view_extents{m_chunks, m_input_lanes};
            static constexpr output_view_extents_t    m_output_view_extents{m_blocks, unroll};

            explicit Kernel(const std::shared_ptr<const Layer>& layer)
                : m_layer(std::move(layer)), m_weights_storage(m_padded_input_size * m_padded_output_size),
                  m_weights(m_weights_storage.data(), m_blocks, m_chunks, unroll, hn::Lanes(Dw())),
                  m_biases_storage(m_padded_output_size), m_biases(m_biases_storage.data(), m_blocks, unroll) {
                const matrix::MatrixView w{m_layer->weights().data(), m_layer->output_size(), m_layer->input_size()};
                const auto               w_t0 = pad(w, m_output_padding, m_input_padding);
                const auto               w_t1 = tile_cols(w_t0, m_input_lanes);
                const auto               w_t2 = hsplit(w_t1, m_blocks);

                const matrix::MatrixView b{m_layer->biases().data(), m_layer->output_size(), 1};
                const auto               b_t0 = pad(b, m_output_padding, 0);

                materialize(w_t2, m_weights_storage.data());
                materialize(b_t0, m_biases_storage.data());
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
            forward(const input_type* HWY_RESTRICT in_ptr, output_type* HWY_RESTRICT out_ptr) const override {
                std::mdspan input{in_ptr, m_input_view_extents};
                std::mdspan output{out_ptr, m_output_view_extents};

                DECLARE_REG_BANK(weights_view_t::static_extent(2), Vout);
                for (extent_type b = 0; b < m_weights.extent(0); ++b) {
                    for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                        GET_REG(u) = hn::Zero(Dout());
                    }
                    for (extent_type c = 0; c < m_weights.extent(1); ++c) {
                        auto in = hn::Load(Din(), &input[c, 0]);
                        for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                            GET_REG(u) =
                                dot(hn::BitCast(Din(), in), hn::Load(Dw(), &m_weights[b, c, u, 0]), GET_REG(u));
                        }
                    }
                    for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                        output[b, u] = hn::ReduceSum(Dout(), GET_REG(u)) + m_biases[b, u];
                    }
                }
            }

          private:
            [[nodiscard]] static Vout
            dot(const Vin a, const Vw b, const Vout acc) {
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

            std::shared_ptr<const Layer> m_layer;
            AlignedVector<weights_type>  m_weights_storage;
            weights_view_t               m_weights{};
            AlignedVector<biases_type>   m_biases_storage{};
            biases_view_t                m_biases{};
        };
    }; // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers
HWY_AFTER_NAMESPACE();

#endif
