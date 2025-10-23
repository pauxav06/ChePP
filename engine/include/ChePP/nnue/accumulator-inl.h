#include "accumulator.h"
#include <cassert>
#include <cstring>
#include <experimental/mdarray>
#include <experimental/mdspan>
#include <hwy/aligned_allocator.h>
#include <vector>

#if defined(CHEPP_ACCUM_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_ACCUM_INL_H_
#undef CHEPP_ACCUM_INL_H_
#else
#define CHEPP_ACCUM_INL_H_
#endif

#include "utils-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::layers::accumulator {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace nn = chepp::nnue::HWY_NAMESPACE;

        using namespace std::experimental;
        using namespace meta;

        template <typename Params>
        struct Layer;

        template <KernelConcept Kernel, TypesConcept Types, DimsConcept Dims>
        struct Layer<std::tuple<Kernel, Types, Dims>> {

            using input_type  = Types::in;
            using output_type = Types::out;

            // transposed
            static constexpr size_t Rows    = Dims::in;
            static constexpr size_t Columns = Dims::out;

            using idx_span_t = std::span<const input_type>;

            using bias_extent_t = extents<size_t, Columns>;
            using bias_array_t =
                mdarray<const output_type, bias_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            using weights_extent_t = extents<size_t, Rows, Columns>;
            using weight_array_t =
                mdarray<const output_type, weights_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            static constexpr weights_extent_t WeightsExtent{Rows, Columns};
            static constexpr bias_extent_t    BiasExtent{Columns};

            weight_array_t m_weights;
            bias_array_t   m_biases;

            void load_weights(const output_type* HWY_RESTRICT weights, const output_type* HWY_RESTRICT biases) {
                m_weights = weight_array_t(WeightsExtent);
                m_biases  = bias_array_t(BiasExtent);
                std::memcpy(m_weights.data(), weights, m_weights.size());
                std::memcpy(m_biases.data(), biases, m_biases.size());
            }

            void forward(const input_type* HWY_RESTRICT idx_ptr,
                         const size_t                   n_idx,
                         output_type* HWY_RESTRICT      out_ptr) const {
                for (size_t c = 0; c < m_biases.extent(0); ++c) {
                    out_ptr[c] = m_biases[c];
                }
                for (size_t i = 0; i < n_idx; ++i) {
                    input_type idx = idx_ptr[i];
                    for (size_t c = 0; c < m_weights.extent(1); ++c) {
                        out_ptr[c] += m_weights[idx, c];
                    }
                }
            }

            void forward_incremental(const input_type* HWY_RESTRICT input_ptr,
                                     const input_type* HWY_RESTRICT added_ptr,
                                     const size_t                   n_added,
                                     const input_type* HWY_RESTRICT removed_ptr,
                                     const size_t                   n_removed,
                                     output_type* HWY_RESTRICT      out_ptr) const {
                for (size_t c = 0; c < m_biases.extent(0); ++c) {
                    out_ptr[c] = input_ptr[c];
                }
                for (size_t i = 0; i < n_added; ++i) {
                    input_type idx = added_ptr[i];
                    for (size_t c = 0; c < m_weights.extent(1); ++c) {
                        out_ptr[c] += m_weights[idx, c];
                    }
                }
                for (size_t i = 0; i < n_removed; ++i) {
                    input_type idx = removed_ptr[i];
                    for (size_t c = 0; c < m_weights.extent(1); ++c) {
                        out_ptr[c] -= m_weights[idx, c];
                    }
                }
            }
        };

        template <TypesConcept Types, DimsConcept Dims, UnrollConcept U>
        struct Layer<std::tuple<Simd, Types, Dims, U>> {

            using input_type  = Types::in;
            using output_type = Types::out;

            static constexpr size_t Rows    = Dims::in;
            static constexpr size_t Columns = Dims::out;
            static constexpr size_t Unroll  = U::value;

            using D   = hn::CappedTag<input_type, Columns>;
            using Vec = hn::VFromD<D>;

            static HWY_LANES_CONSTEXPR size_t Lanes  = hn::Lanes(D());
            static HWY_LANES_CONSTEXPR size_t Chunks = Columns / (Lanes * Unroll);

            using output_extent_t =
                extents<size_t, nn::extent_if_constexpr_v<Chunks>, Unroll, nn::extent_if_constexpr_v<Lanes>>;
            using const_output_view   = mdspan<const input_type, output_extent_t>;
            using output_view_t       = mdspan<input_type, output_extent_t>;
            using sparse_input_span_t = std::span<const input_type>;

            using weights_extent_t = extents<size_t,
                                             Rows,
                                             nn::extent_if_constexpr_v<Chunks>,
                                             nn::extent_if_constexpr_v<Unroll>,
                                             nn::extent_if_constexpr_v<Lanes>>;

            using weight_array_t =
                mdarray<const output_type, weights_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            using bias_extent_t = extents<size_t,
                                          nn::extent_if_constexpr_v<Chunks>,
                                          nn::extent_if_constexpr_v<Unroll>,
                                          nn::extent_if_constexpr_v<Lanes>>;
            using bias_array_t =
                mdarray<const output_type, bias_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            static HWY_LANES_CONSTEXPR weights_extent_t WeightsExtent{Rows, Chunks, Unroll, Lanes};
            static HWY_LANES_CONSTEXPR bias_extent_t    BiasExtent{Chunks, Unroll, Lanes};
            static HWY_LANES_CONSTEXPR output_extent_t  OutputView{Chunks, Unroll, Lanes};

            weight_array_t m_weights;
            bias_array_t   m_biases;

            void load_weights(const output_type* HWY_RESTRICT weights, const output_type* HWY_RESTRICT biases) {
                m_weights = weight_array_t(WeightsExtent);
                m_biases  = bias_array_t(BiasExtent);
                std::memcpy(m_weights.data(), weights, m_weights.size());
                std::memcpy(m_biases.data(), biases, m_biases.size());
            }

            void forward(const input_type* HWY_RESTRICT idx_ptr,
                         const size_t                   n_idx,
                         output_type* HWY_RESTRICT      out_ptr) const {

                output_view_t       out{out_ptr, OutputView};
                sparse_input_span_t indices{idx_ptr, n_idx};

                for (size_t c = 0; c < Chunks; ++c) {
                    chepp::nnue::HWY_NAMESPACE::RegisterBank<Unroll, Vec>::run(
                        [&](const size_t u) { return hn::Load(D(), &m_biases[c, u, 0]); },
                        [&](auto get_reg, auto set_reg) {
                            for (const auto idx : indices) {
                                for (size_t u = 0; u < Unroll; ++u) {
                                    set_reg(u, get_reg(u) + hn::Load(D(), &m_weights[idx, c, u, 0]));
                                }
                            }
                            for (size_t u = 0; u < Unroll; ++u) hn::Store(get_reg(u), D(), &out[c, u, 0]);
                        });
                }
            }

            void forward_incremental(const input_type* HWY_RESTRICT input_ptr,
                                     const input_type* HWY_RESTRICT added_ptr,
                                     const size_t                   n_added,
                                     const input_type* HWY_RESTRICT removed_ptr,
                                     const size_t                   n_removed,
                                     output_type* HWY_RESTRICT      out_ptr) const {

                const_output_view   curr{input_ptr, OutputView};
                output_view_t       out{out_ptr, OutputView};
                sparse_input_span_t added{added_ptr, n_added};
                sparse_input_span_t removed{removed_ptr, n_removed};

                for (size_t c = 0; c < Chunks; ++c) {
                    chepp::nnue::HWY_NAMESPACE::RegisterBank<Unroll, Vec>::run(
                        [&](const size_t u) { return hn::Load(D(), &curr[c, u, 0]); },
                        [&](auto get_reg, auto set_reg) {
                            for (const auto idx : added) {
                                for (size_t u = 0; u < Unroll; ++u) {
                                    set_reg(u, get_reg(u) + hn::Load(D(), &m_weights[idx, c, u, 0]));
                                }
                            }
                            for (const auto idx : removed) {
                                for (size_t u = 0; u < Unroll; ++u) {
                                    set_reg(u, get_reg(u) - hn::Load(D(), &m_weights[idx, c, u, 0]));
                                }
                            }
                            for (size_t u = 0; u < Unroll; ++u) hn::Store(get_reg(u), D(), &out[c, u, 0]);
                        });
                }
            }
        };
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::accumulator
HWY_AFTER_NAMESPACE();

#endif
