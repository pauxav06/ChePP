#include "accumulator.h"
#include "matrix.h"
#include <cassert>
#include <cstring>
#include <hwy/aligned_allocator.h>
#include <hwy/cache_control.h>
#include <mdspan>
#include <vector>

#if defined(CHEPP_ACCUM_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_ACCUM_INL_H_
#undef CHEPP_ACCUM_INL_H_
#else
#define CHEPP_ACCUM_INL_H_
#endif

#include "layer-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;

        template <typename IdxT, std::size_t IS, typename OutT, std::size_t OS, auto cfg>
        struct Kernel<AccumulatorLayer<IdxT, IS, OutT, OS>, cfg> final : AccumulatorLayer<IdxT, IS, OutT, OS>::IKernel {
            using Layer      = AccumulatorLayer<IdxT, IS, OutT, OS>;
            using index_type = Layer::index_type;
            using value_type = Layer::value_type;

            using weights_extents_t = std::extents<std::size_t, Layer::input_size(), Layer::output_size()>;
            using biases_extents_t  = std::extents<std::size_t, Layer::output_size()>;
            using weights_view_t    = std::mdspan<const value_type, weights_extents_t>;
            using biases_view_t     = std::mdspan<const value_type, biases_extents_t>;

            explicit Kernel(const std::shared_ptr<Layer>& layer)
                : m_layer(layer), m_weights(std::data(m_layer->weights()), weights_extents_t{}),
                  m_biases(std::data(m_layer->biases()), biases_extents_t{}) {
            }

            void
            forward(const index_type* HWY_RESTRICT idx,
                    const std::size_t              nb_idx,
                    value_type* HWY_RESTRICT       out_ptr) const override {
                std::memcpy(out_ptr, std::data(m_layer->biases()), std::size(m_layer->biases()) * sizeof(value_type));
                for (std::size_t i = 0; i < nb_idx; ++i) {
                    for (std::size_t j = 0; j < Layer::output_size(); ++j) {
                        out_ptr[j] += m_weights[idx[i], j];
                    }
                }
            }

            void
            forward_incremental(const value_type* HWY_RESTRICT input_ptr,
                                const index_type* HWY_RESTRICT added,
                                const std::size_t              nb_added,
                                const index_type* HWY_RESTRICT removed,
                                const std::size_t              nb_removed,
                                value_type* HWY_RESTRICT       out_ptr) const override {
                std::memcpy(out_ptr, input_ptr, Layer::output_size() * sizeof(value_type));
                for (std::size_t i = 0; i < nb_added; ++i) {
                    for (std::size_t j = 0; j < Layer::output_size(); ++j) {
                        out_ptr[j] += m_weights[added[i], j];
                    }
                }
                for (std::size_t i = 0; i < nb_removed; ++i) {
                    for (std::size_t j = 0; j < Layer::output_size(); ++j) {
                        out_ptr[j] -= m_weights[removed[i], j];
                    }
                }
            }

          private:
            std::shared_ptr<Layer> m_layer;
            weights_view_t         m_weights;
            biases_view_t          m_biases;
        };

        template <typename IdxT, std::size_t IS, typename OutT, std::size_t OS, AccumulatorSimd cfg>
        struct Kernel<AccumulatorLayer<IdxT, IS, OutT, OS>, cfg> final : AccumulatorLayer<IdxT, IS, OutT, OS>::IKernel {
            using Layer      = AccumulatorLayer<IdxT, IS, OutT, OS>;
            using index_type = Layer::index_type;
            using value_type = Layer::value_type;

            static constexpr std::size_t unroll = cfg.unroll;

            using D   = hn::ScalableTag<value_type>;
            using Vec = hn::VFromD<D>;

            HWY_STATIC_CONSTEXPR std::size_t m_lanes{hn::Lanes(D())};
            HWY_STATIC_CONSTEXPR std::size_t m_padded_output_size{pad_up(Layer::output_size(), m_lanes* unroll)};
            HWY_STATIC_CONSTEXPR std::size_t m_output_padding{m_padded_output_size - Layer::output_size()};
            HWY_STATIC_CONSTEXPR std::size_t m_chunks{m_padded_output_size / (m_lanes * unroll)};

            using output_extents_t =
                std::extents<std::size_t, HWY_CONSTEXPR_EXT(m_chunks), unroll, HWY_CONSTEXPR_EXT(m_lanes)>;
            using output_view_t     = std::mdspan<value_type, output_extents_t>;
            using input_extents_t   = output_extents_t;
            using input_view_t      = std::mdspan<const value_type, input_extents_t>;
            using weights_extents_t = std::extents<std::size_t,
                                                   Layer::input_size(),
                                                   HWY_CONSTEXPR_EXT(m_chunks),
                                                   unroll,
                                                   HWY_CONSTEXPR_EXT(m_lanes)>;
            using weight_view_t     = std::mdspan<const value_type, weights_extents_t>;
            using bias_extents_t =
                std::extents<std::size_t, HWY_CONSTEXPR_EXT(m_chunks), unroll, HWY_CONSTEXPR_EXT(m_lanes)>;
            using bias_view_t = std::mdspan<const value_type, bias_extents_t>;

            HWY_STATIC_CONSTEXPR output_extents_t m_output_extents{m_chunks, unroll, m_lanes};

            explicit Kernel(const std::shared_ptr<Layer>& layer)
                : m_layer(layer), m_weights_storage(Layer::input_size() * m_padded_output_size),
                  m_weights(std::data(m_weights_storage), Layer::input_size(), m_chunks, unroll, m_lanes),
                  m_biases_storage(m_padded_output_size),
                  m_biases(std::data(m_biases_storage), m_chunks, unroll, m_lanes) {
                const matrix::MatrixView w{std::data(m_layer->weights()), Layer::input_size(), Layer::output_size()};
                const auto               w_t0 = pad(w, 0, m_output_padding);

                const matrix::MatrixView b{std::data(m_layer->biases()), 1, Layer::output_size()};
                const auto               b_t0 = pad(b, 0, m_output_padding);

                materialize(w_t0, std::data(m_weights_storage));
                materialize(b_t0, std::data(m_biases_storage));
            }

            [[nodiscard]] std::size_t
            padding() const noexcept {
                return m_output_padding;
            }

            void
            forward(const index_type* HWY_RESTRICT idx_ptr,
                    const std::size_t              n_idx,
                    value_type* HWY_RESTRICT       out_ptr) const override {

                output_view_t out{out_ptr, m_output_extents};

                DECLARE_REG_BANK(unroll, Vec)
                for (std::size_t c{0}; c < out.extent(0); ++c) {
                    for (std::size_t u{0}; u < out.extent(1); ++u) {
                        *regs[u] = hn::Load(D(), &m_biases[c, u, 0]);
                    }
                    for (std::size_t i{0}; i < n_idx; ++i) {
                        if (i + 1 < n_idx) {
                            hwy::Prefetch(&m_weights[idx_ptr[i + 1], c, 0, 0]);
                        }
                        for (std::size_t u{0}; u < out.extent(1); ++u) {
                            *regs[u] += hn::Load(D(), &m_weights[idx_ptr[i], c, u, 0]);
                        }
                    }
                    for (std::size_t u{0}; u < out.extent(1); ++u) {
                        hn::Store(*regs[u], D(), &out[c, u, 0]);
                    }
                }
            }

            void
            forward_incremental(const value_type* HWY_RESTRICT input_ptr,
                                const index_type* HWY_RESTRICT added_ptr,
                                const size_t                   n_added,
                                const index_type* HWY_RESTRICT removed_ptr,
                                const size_t                   n_removed,
                                value_type* HWY_RESTRICT       out_ptr) const override {

                input_view_t  input{input_ptr, m_output_extents};
                output_view_t out{out_ptr, m_output_extents};

                DECLARE_REG_BANK(unroll, Vec)
                for (std::size_t c{0}; c < out.extent(0); ++c) {
                    for (std::size_t u{0}; u < out.extent(1); ++u) {
                        *regs[u] = hn::Load(D(), &input[c, u, 0]);
                    }
                    for (std::size_t i{0}; i < n_added; ++i) {
                        for (std::size_t u{0}; u < out.extent(1); ++u) {
                            *regs[u] += hn::Load(D(), &m_weights[added_ptr[i], c, u, 0]);
                        }
                    }
                    for (std::size_t i{0}; i < n_removed; ++i) {
                        for (std::size_t u{0}; u < out.extent(1); ++u) {
                            *regs[u] -= hn::Load(D(), &m_weights[removed_ptr[i], c, u, 0]);
                        }
                    }
                    for (std::size_t u{0}; u < out.extent(1); ++u) {
                        hn::Store(*regs[u], D(), &out[c, u, 0]);
                    }
                }
            }

            std::shared_ptr<Layer>         m_layer;
            hwy::AlignedVector<value_type> m_weights_storage;
            weight_view_t                  m_weights;
            hwy::AlignedVector<value_type> m_biases_storage;
            bias_view_t                    m_biases;
        };
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers
HWY_AFTER_NAMESPACE();

#endif
