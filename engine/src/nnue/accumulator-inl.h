#include "accumulator.h"
#include "matrix.h"
#include <cassert>
#include <cstring>
#include <hwy/aligned_allocator.h>
#include <hwy/cache_control.h>

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

        template <typename IdxT, std::size_t IS, typename OutT, std::size_t OS, default_config_t cfg>
            requires(std::is_same_v<default_config_t, decltype(cfg)>)
        struct Kernel<Accumulator<IdxT, IS, OutT, OS>, cfg> final : Accumulator<IdxT, IS, OutT, OS>::ikernel_t {
            using accum_t = Accumulator<IdxT, IS, OutT, OS>;
            using layer_t = accum_t::layer_t;
            using base_t  = accum_t::ikernel_t;

            using base_t::base_t;
            using base_t::layer;

            using index_t = accum_t::index_t;
            using value_t = accum_t::value_t;

            using weights_extents_t = std::extents<std::size_t, accum_t::input_size_v, accum_t::output_size_v>;
            using weights_t =
                stdx::mdarray<const value_t, weights_extents_t, std::layout_right, hwy::AlignedVector<value_t>>;

            using biases_extents_t = std::extents<std::size_t, accum_t::output_size_v>;
            using biases_t =
                stdx::mdarray<const value_t, biases_extents_t, std::layout_right, hwy::AlignedVector<value_t>>;

            [[nodiscard]] std::string
            name() const noexcept override {
                return std::format("default,target={}", hwy::TargetName(HWY_TARGET));
            }

            explicit Kernel(const std::shared_ptr<layer_t>& l) noexcept : base_t(l) {
                std::ranges::copy(layer().weights(), std::ranges::begin(m_weights.container()));
                std::ranges::copy(layer().biases(), std::ranges::begin(m_biases.container()));
            }

            void
            forward(const index_t* HWY_RESTRICT idx,
                    const std::size_t           nb_idx,
                    value_t* HWY_RESTRICT       out_ptr) const noexcept override {
                std::memcpy(out_ptr, std::data(m_biases), std::size(m_biases) * sizeof(value_t));
                for (std::size_t i = 0; i < nb_idx; ++i) {
                    for (std::size_t j = 0; j < accum_t::output_size_v; ++j) {
                        out_ptr[j] += m_weights[idx[i], j];
                    }
                }
            }

            void
            forward_incremental(const value_t* HWY_RESTRICT input_ptr,
                                const index_t* HWY_RESTRICT added,
                                const std::size_t           nb_added,
                                const index_t* HWY_RESTRICT removed,
                                const std::size_t           nb_removed,
                                value_t* HWY_RESTRICT       out_ptr) const noexcept override {
                std::memcpy(out_ptr, input_ptr, accum_t::output_size_v * sizeof(value_t));
                for (std::size_t i = 0; i < nb_added; ++i) {
                    for (std::size_t j = 0; j < accum_t::output_size_v; ++j) {
                        out_ptr[j] += m_weights[added[i], j];
                    }
                }
                for (std::size_t i = 0; i < nb_removed; ++i) {
                    for (std::size_t j = 0; j < accum_t::output_size_v; ++j) {
                        out_ptr[j] -= m_weights[removed[i], j];
                    }
                }
            }

          private:
            weights_t m_weights;
            biases_t  m_biases;
        };

        template <typename IdxT, std::size_t IS, typename OutT, std::size_t OS, AccumulatorSimd cfg>
            requires(std::is_same_v<AccumulatorSimd, decltype(cfg)>)
        struct Kernel<Accumulator<IdxT, IS, OutT, OS>, cfg> final : Accumulator<IdxT, IS, OutT, OS>::ikernel_t {
            using accum_t = Accumulator<IdxT, IS, OutT, OS>;
            using layer_t = accum_t::layer_t;
            using base_t  = accum_t::ikernel_t;

            using base_t::base_t;
            using base_t::layer;

            using index_t = accum_t::index_t;
            using value_t = accum_t::value_t;

            static constexpr std::size_t unroll = cfg.unroll;

            using D   = hn::ScalableTag<value_t>;
            using Vec = hn::VFromD<D>;

            HWY_STATIC_CONSTEXPR std::size_t lanes{hn::Lanes(D())};
            HWY_STATIC_CONSTEXPR std::size_t padded_output_size{pad_up(accum_t::output_size_v, lanes* unroll)};
            HWY_STATIC_CONSTEXPR std::size_t output_padding{padded_output_size - accum_t::output_size_v};
            HWY_STATIC_CONSTEXPR std::size_t chunks{padded_output_size / (lanes * unroll)};

            using output_extents_t =
                std::extents<std::size_t, HWY_CONSTEXPR_EXT(chunks), unroll, HWY_CONSTEXPR_EXT(lanes)>;
            using output_view_t     = std::mdspan<value_t, output_extents_t>;
            using input_extents_t   = output_extents_t;
            using input_view_t      = std::mdspan<const value_t, input_extents_t>;
            using weights_extents_t = std::extents<std::size_t,
                                                   accum_t::input_size_v,
                                                   HWY_CONSTEXPR_EXT(chunks),
                                                   unroll,
                                                   HWY_CONSTEXPR_EXT(lanes)>;
            using weights_t =
                stdx::mdarray<const value_t, weights_extents_t, std::layout_right, hwy::AlignedVector<value_t>>;

            using bias_extents_t =
                std::extents<std::size_t, HWY_CONSTEXPR_EXT(chunks), unroll, HWY_CONSTEXPR_EXT(lanes)>;
            using bias_t = stdx::mdarray<const value_t, bias_extents_t, std::layout_right, hwy::AlignedVector<value_t>>;

            [[nodiscard]] std::string
            name() const noexcept override {
                return std::format("simd,target={},unroll={}", hwy::TargetName(HWY_TARGET), unroll);
            }

            [[nodiscard]] std::size_t
            padding() const noexcept override {
                return output_padding;
            }

            explicit Kernel(const std::shared_ptr<layer_t>& l) noexcept
                : base_t(l), m_weights(accum_t::input_size_v, chunks, unroll, lanes), m_biases(chunks, unroll, lanes) {
                const matrix::MatrixView w{std::data(layer().weights()), accum_t::input_size_v, accum_t::output_size_v};
                const auto               w_t0 = pad(w, 0, output_padding);

                const matrix::MatrixView b{std::data(layer().biases()), 1, accum_t::output_size_v};
                const auto               b_t0 = pad(b, 0, output_padding);

                materialize(w_t0, std::data(m_weights));
                materialize(b_t0, std::data(m_biases));
            }

            void
            forward(const index_t* HWY_RESTRICT idx_ptr,
                    const std::size_t           n_idx,
                    value_t* HWY_RESTRICT       out_ptr) const noexcept override {
                output_view_t out{out_ptr, m_biases.extents()};

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
            forward_incremental(const value_t* HWY_RESTRICT input_ptr,
                                const index_t* HWY_RESTRICT added_ptr,
                                const size_t                n_added,
                                const index_t* HWY_RESTRICT removed_ptr,
                                const size_t                n_removed,
                                value_t* HWY_RESTRICT       out_ptr) const noexcept override {

                input_view_t  input{input_ptr, m_biases.extents()};
                output_view_t out{out_ptr, m_biases.extents()};

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

          private:
            weights_t m_weights;
            bias_t    m_biases;
        };
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers
HWY_AFTER_NAMESPACE();

#endif
