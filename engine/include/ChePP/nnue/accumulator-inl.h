#include "utils.h"

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
namespace chepp::nnue::layers::accum {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace nn = chepp::nnue::HWY_NAMESPACE;
        using namespace std::experimental;

        template <typename InT, size_t Rows_, typename OutT, size_t Columns_>
        struct ScalarKernel {
            using input_type  = InT;
            using output_type = OutT;

            static constexpr size_t Rows    = Rows_;
            static constexpr size_t Columns = Columns_;

            using idx_span_t = std::span<const input_type>;

            using bias_extent_t = extents<size_t, Columns>;
            using bias_array_t =
                mdarray<const output_type, bias_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            using weights_extent_t = extents<size_t, Rows, Columns>;
            using weight_array_t =
                mdarray<const output_type, weights_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            static constexpr weights_extent_t WeightsExtent{Rows, Columns};

            weight_array_t m_weights;

            ScalarKernel() = default;

            void load_weights(const output_type* weights, const output_type* biases) {
                m_weights = weight_array_t(WeightsExtent);
                std::memcpy(m_weights.data(), weights, Rows * Columns * sizeof(output_type));
            }

            void forward(const output_type* in, const input_type* indices, size_t n, output_type* out,
                         bool add = true) const {
                for (size_t c = 0; c < Columns; ++c) {
                    out[c] += in[c];
                }
                for (size_t i = 0; i < n; ++i) {
                    input_type idx = indices[i];
                    for (size_t c = 0; c < Columns; ++c) {
                        out[c] += m_weights[idx, c];
                    }
                }
            }
        };

        template <typename T, size_t Rows_, size_t Columns_, typename IdxT = size_t>
        struct SIMDKernel {
            static constexpr size_t Unroll  = 8;
            static constexpr size_t Rows    = Rows_;
            static constexpr size_t Columns = Columns_;

            using D   = hn::ScalableTag<T>;
            using Vec = hn::VFromD<D>;

            static HWY_LANES_CONSTEXPR size_t Lanes  = hn::Lanes(D());
            static HWY_LANES_CONSTEXPR size_t Chunks = Columns / (Lanes * Unroll);

            using vec_extent_t =
                extents<size_t, nn::extent_if_constexpr_v<Chunks>, Unroll, nn::extent_if_constexpr_v<Lanes>>;
            using const_vec_span_t    = mdspan<const T, vec_extent_t>;
            using vec_span_t          = mdspan<T, vec_extent_t>;
            using sparse_input_span_t = std::span<const IdxT>;

            using weights_extent_t = extents<size_t, Rows, nn::extent_if_constexpr_v<Chunks>,
                                             nn::extent_if_constexpr_v<Unroll>, nn::extent_if_constexpr_v<Lanes>>;

            using weight_array_t = mdarray<const T, weights_extent_t, layout_right, hwy::AlignedVector<T>>;

            static HWY_LANES_CONSTEXPR weights_extent_t WeightsExtent{Rows, Chunks, Unroll, Lanes};
            static HWY_LANES_CONSTEXPR vec_extent_t     VecExtent{Chunks, Unroll, Lanes};

            weight_array_t m_weights;

            SIMDKernel() = default;

            void load_weights(const T* weights) {
                m_weights = weight_array_t(WeightsExtent);
                std::memcpy(m_weights.data(), weights, Rows * Columns * sizeof(T));
            }

            HWY_NOINLINE void forward(T* HWY_RESTRICT vec, const IdxT* HWY_RESTRICT idx_ptr, size_t n,
                                      T* out_ptr) const {
                const_vec_span_t    in{vec, VecExtent};
                vec_span_t          out{out_ptr, VecExtent};
                sparse_input_span_t indices{idx_ptr, idx_ptr + n};

                for (size_t c = 0; c < Chunks; ++c) {
                    chepp::nnue::HWY_NAMESPACE::RegisterBank<Unroll, Vec>::run(
                        [&](const size_t u) { return hn::Load(D(), &in[c, u, 0]); },
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
        };
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::accum
HWY_AFTER_NAMESPACE();

#endif
