#include "hwy/auto_tune.h"
#include "layers.h"
#include "simd/traits.h"
#include "transformations.hpp"
#include "utils.h"

#include <hwy/aligned_allocator.h>
#include <mdspan/mdarray.hpp>


#if defined(CHEPP_AFFINE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_AFFINE_INL_H_
#undef CHEPP_AFFINE_INL_H_
#else
#define CHEPP_AFFINE_INL_H_
#endif

#include "utils-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers::affine
{
    namespace HWY_NAMESPACE
    {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace nn = chepp::nnue::HWY_NAMESPACE;

        template <Kernels K>
        struct Params;

        template <Kernels K, Params<K> P>
        struct Kernel;

        template<>
        struct Params<Kernels::Scalar>
        {
            size_t Rows;
            size_t Cols;
        };

        template <Params<Kernels::Scalar> P>
        struct Kernel<Kernels::Scalar, P>
        {
            static constexpr size_t Rows = P.Rows;
            static constexpr size_t Cols = P.Cols;
            using weights_extent_t = Kokkos::extents<size_t, Rows, Cols>;
            using biases_extent_t  = Kokkos::extents<size_t, Cols>;
            using weights_t        = Kokkos::mdspan<const int8_t, weights_extent_t>;
            using bias_t           = Kokkos::mdspan<const int32_t, biases_extent_t>;
            weights_extent_t m_weights_extent{};
            biases_extent_t  m_biases_extent{};

            Kernel()
                : m_w_ptr(hwy::AllocateAligned<int8_t>(Rows * Cols)),
                  m_b_ptr(hwy::AllocateAligned<int32_t>(Cols)), m_weights(m_w_ptr.get(), m_weights_extent),
                  m_biases(m_b_ptr.get(), m_biases_extent)
            {
            }

            void load_weights(const int8_t* w, const int32_t* b)
            {
                std::memcpy(m_w_ptr.get(), w, sizeof(int8_t) * Rows * Cols);
                std::memcpy(m_b_ptr.get(), b, sizeof(int32_t) * Cols);
            }

            void forward(const int8_t* input, int32_t* output)
            {
                for (size_t r = 0; r < Rows; ++r)
                {
                    int32_t acc = 0;
                    for (size_t c = 0; c < Cols; ++c)
                        acc += static_cast<int32_t>(m_weights[r, c]) * static_cast<int32_t>(input[c]);
                    output[r] = acc + m_biases[r];
                }
            }

            hwy::AlignedFreeUniquePtr<int8_t[]>  m_w_ptr;
            hwy::AlignedFreeUniquePtr<int32_t[]> m_b_ptr;
            weights_t                            m_weights;
            bias_t                               m_biases;
        };

        struct VNNIKernelParams
        {
            size_t Unroll                = 8;
            bool   NativeSumOfMulQuadAcc = false;
            bool   NativeSumOfMulPairAcc = false;
        };

        template <size_t Rows_, size_t Columns_, VNNIKernelParams Params_>
        struct VNNIKernel
        {
            static constexpr auto Unroll = Params_.Unroll;

            static constexpr size_t Columns = Columns_;
            static_assert(Columns % (4 * Unroll) == 0);

            using Di8    = hn::ScalableTag<int8_t>;
            using Du8    = hn::RebindToUnsigned<Di8>;
            using Di16   = hn::RepartitionToWide<Di8>;
            using Di32   = hn::RepartitionToWide<Di16>;
            using Veci8  = hn::VFromD<Di8>;
            using Vecu8  = hn::VFromD<Du8>;
            using Veci16 = hn::VFromD<Di16>;
            using Veci32 = hn::VFromD<Di32>;

            static constexpr size_t Chunks = Columns / 4 / Unroll;
            // Lanes might or might not be constexpr :) We must not assume they are
            static HWY_LANES_CONSTEXPR size_t D8Lanes  = hn::Lanes(Di8());
            static HWY_LANES_CONSTEXPR size_t D32Lanes = hn::Lanes(Di32());

            static HWY_LANES_CONSTEXPR size_t Rows       = utils::pad_up(Rows_, D32Lanes);
            static HWY_LANES_CONSTEXPR size_t RowPadding = Rows - Rows_;
            static HWY_LANES_CONSTEXPR size_t Blocks     = Rows / D32Lanes;


            // we make extensive internal use of mdspan/mdarray in multiple layer implementations.
            // this makes indexing a lot more flexible, safe and efficient as constexpr extents will always be
            // correctly precomputed. Unfortunatly it is not yet widely implemented by all compilers so we just pull
            // it from the GitHub repo.
            using weights_extent_t = Kokkos::extents<size_t,
                nn::extent_if_constexpr_v<Blocks>,
                Chunks,
                Unroll,
                nn::extent_if_constexpr_v<D8Lanes>>;
            using biases_extent_t  = Kokkos::extents<size_t,
                nn::extent_if_constexpr_v<Blocks>,
                nn::extent_if_constexpr_v<D32Lanes>>;

            static HWY_LANES_CONSTEXPR weights_extent_t WeightsExtent{Blocks, Chunks, Unroll, D8Lanes};
            static HWY_LANES_CONSTEXPR biases_extent_t  BiasesExtent{Blocks, D32Lanes};

            using input_extent_t = Kokkos::extents<size_t, Chunks, Unroll, 4>;
            static HWY_LANES_CONSTEXPR input_extent_t InputExtent{};
            using padded_output_extent_t                                         = biases_extent_t;
            static HWY_LANES_CONSTEXPR padded_output_extent_t PaddedOutputExtent = BiasesExtent;

            using input_t         = Kokkos::mdspan<const int8_t, input_extent_t>;
            using weights_array_t = Kokkos::Experimental::mdarray<const int8_t, weights_extent_t, Kokkos::layout_right,
                                                                  hwy::AlignedVector<int8_t>>;
            using biases_array_t  = Kokkos::Experimental::mdarray<const int32_t, biases_extent_t, Kokkos::layout_right,
                                                                  hwy::AlignedVector<int32_t>>;
            using output_array_t  = Kokkos::Experimental::mdarray<int32_t, padded_output_extent_t, Kokkos::layout_right,
                                                                  hwy::AlignedVector<int32_t>>;

            weights_array_t m_weights;
            biases_array_t  m_biases;

            static HWY_LANES_CONSTEXPR bool valid_split = []()
            {
                size_t t = 1;
                for (int r = 0; r < weights_extent_t::rank(); ++r) t *= WeightsExtent.extent(r);
                return t == Rows * Columns;
            }();

            HWY_STATIC_ASSERT(valid_split, "Weights split does not match total size");

            // for every target, both versions will be compiled, the best one will be chosen at runtime
            static constexpr auto SumOfMulQuadAcc = [](const Vecu8 a, const Veci8 b, const Veci32 acc)
            {
                if constexpr (Params_.NativeSumOfMulQuadAcc)
                    return hn::SumOfMulQuadAccumulate(Di32(), a, b, acc); // VNNI / ARM & RVV Dot
                else if constexpr (Params_.NativeSumOfMulPairAcc || true)
                {
                    // This version can overflow but is much faster that the exact emulation of SumOfMulQuadAccumulate
                    // NEON 8, x86 >= SSSE3
                    const Veci16 sum0 = hn::SatWidenMulPairwiseAdd(Di16(), a, b);
                    const Veci32 sum1 = hn::WidenMulPairwiseAdd(Di32(), sum0, hn::Set(Di16(), 1));
                    return hn::Add(acc, sum1);
                }
            };

            void load_weights(const hn::TFromD<Di8>* w, const hn::TFromD<Di32>* b)
            {
                m_weights = weights_array_t{WeightsExtent};
                m_biases = biases_array_t{BiasesExtent};
                AnyMatrixView<hn::TFromD<Di8>> weightsView{MatrixView{w, Rows_, Columns_, Columns_}};
                weightsView = tile_cols(weightsView, 4);
                weightsView = pad(weightsView, RowPadding * 4, 0);
                weightsView = tile_cols(weightsView, D8Lanes);

                AnyMatrixView<hn::TFromD<Di32>> biasesView{MatrixView{b, Rows_, 1, 1}};
                biasesView = pad(biasesView, 0, RowPadding);

                weightsView.materialize(m_weights.data());
                biasesView.materialize(m_biases.data());
            }

            template <size_t N, typename GetFunc>
            static HWY_INLINE Veci32 reduce_tree(GetFunc&& get)
            {
                if constexpr (N == 1) return get(0);
                else
                {
                    constexpr size_t Half  = N / 2;
                    const Veci32     left  = reduce_tree<Half>([&](const size_t i) { return get(i); });
                    const Veci32     right = reduce_tree<N - Half>([&](const size_t i) { return get(i + Half); });
                    return left + right;
                }
            }

            void forward(const int8_t* HWY_RESTRICT input_ptr, int32_t* HWY_RESTRICT output_ptr) const
            {
                thread_local output_array_t tmp_out{PaddedOutputExtent};
                const input_t               input{input_ptr, InputExtent};

                for (auto b = 0; b < m_weights.extent(0); ++b)
                {
                    chepp::nnue::HWY_NAMESPACE::RegisterBank<Unroll, Veci32>::run(
                        [&](const size_t u) { return u == 0 ? hn::Load(Di32(), &m_biases[b, 0]) : hn::Zero(Di32()); },
                        [&](auto get_reg, auto set_reg)
                        {
                            for (auto c = 0; c < m_weights.extent(1); ++c)
                            {
                                for (auto u = 0; u < m_weights.extent(2); ++u)
                                {
                                    Veci32        reg    = get_reg(u);
                                    const int32_t in_val = *reinterpret_cast<const int32_t*>(&input[c, u, 0]);
                                    reg                  = SumOfMulQuadAcc(hn::BitCast(Du8(), hn::Set(Di32(), in_val)),
                                                                           hn::Load(Di8(), &m_weights[b, c, u, 0]), reg);
                                    set_reg(u, reg);
                                }

                                const Veci32 out = reduce_tree<Unroll>([&](size_t i) { return get_reg(i); });
                                hn::Store(out, Di32(), &tmp_out[b, 0]);
                            }
                        });


                    // final write into the non-padded buffer
                    std::memcpy(output_ptr, tmp_out.data(), Rows_ * sizeof(int32_t));
                }
            }
        };
    }; // namespace HWY_NAMESPACE

} // namespace chepp::nnue::affine

HWY_AFTER_NAMESPACE();

#endif