#include "affine.h"
#include "matrix.h"
#include "utils.h"
#include <hwy/auto_tune.h>
#include <iostream>

#include <experimental/mdarray>
#include <experimental/mdspan>
#include <hwy/aligned_allocator.h>

#if defined(CHEPP_AFFINE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_AFFINE_INL_H_
#undef CHEPP_AFFINE_INL_H_
#else
#define CHEPP_AFFINE_INL_H_
#endif

#include "utils-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::layers::affine {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace nn = chepp::nnue::HWY_NAMESPACE;

        using namespace std::experimental;
        using namespace meta;

        template <Kernels K, Params<K> P>
        struct Layer;

        template <Params<Kernels::Scalar> P>
            requires(std::is_same_v<scalar_t<P.types.in>, int8_t> && std::is_same_v<scalar_t<P.types.out>, int32_t>)
        struct Layer<Kernels::Scalar, P> {
            static constexpr auto params      = P;
            using input_type                  = scalar_t<params.types.in>;
            using output_type                 = scalar_t<params.types.out>;
            using extent_type                 = size_t;
            static constexpr extent_type Cols = params.dims.in;
            static constexpr extent_type Rows = params.dims.out;

            using weights_extent_t = extents<extent_type, Rows, Cols>;
            using biases_extent_t  = extents<extent_type, Cols>;
            using weight_array_t   = mdarray<const input_type, weights_extent_t, layout_right, hwy::AlignedVector<input_type>>;
            using bias_array_t     = mdarray<const output_type, biases_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            weights_extent_t m_weights_extent;
            biases_extent_t  m_biases_extent;

            static constexpr weights_extent_t WeightsExtent{Rows, Cols};
            static constexpr biases_extent_t  BiasesExtent{Cols};

            void load_weights(const input_type* HWY_RESTRICT w, const output_type* HWY_RESTRICT b) {
                m_weights = weight_array_t{WeightsExtent};
                m_biases  = bias_array_t{BiasesExtent};
                std::memcpy(m_weights.data(), w, m_weights.size());
                std::memcpy(m_biases.data(), b, m_biases.size());
            }

            void forward(const input_type* input, output_type* output) const {
                for (extent_type r = 0; r < Rows; ++r) {
                    output_type acc = 0;
                    for (extent_type c = 0; c < Cols; ++c)
                        acc += static_cast<output_type>(m_weights[r, c]) * static_cast<output_type>(input[c]);
                    output[r] = acc + m_biases[r];
                }
            }

          private:
            weight_array_t m_weights;
            bias_array_t   m_biases;
        };

        // ============================================================================
        // SIMD Kernel: int8 -> int32 Affine Layer using VNNI / Dot Product Instructions
        // ----------------------------------------------------------------------------
        // This kernel performs an affine transformation:
        //
        //     y = W * x + b
        //
        // where:
        //   - x ∈ M(i8)[Rows_ * 1]     : input vector
        //   - W ∈ M(i8)[Rows_ * Cols_] : weight matrix
        //   - b ∈ M(i32)[Rows * 1]     : bias vector
        //   - y ∈ M(i32)[Rows * 1]     : output vector
        //
        // To leverage SIMD efficiently, the kernel performs several transformations
        // on the weight matrix and input vector:
        //
        //  1 Pad the input dimension (Cols_) to the next multiple of Pack * Unroll:
        //      Pack = sizeof(int32_t)/sizeof(int8_t) = 4
        //      Unroll = loop unroll factor
        //    This ensures that input elements can be grouped in 4s for VNNI/Dotprod.
        //
        //  2 Tiled transposition of weights:
        //      - First, group columns in blocks of 4 (Pack) to align with lane-wise
        //        dot-product instructions.
        //      - Second, perform another tiling to align with the SIMD register width
        //        (D8Lanes). This ensures contiguous memory accesses during vector loads.
        //
        //  3. Forward pass:
        //      For each block b of output rows and each chunk c of input columns:
        //          acc[u] = SumOfMulQuadAcc(packed_input[c,u], weights[b,c,u], acc[u])
        //
        //      SumOfMulQuadAcc implements the 4×4 lane-wise multiply-accumulate used in VNNI/Neon Dotprod:
        //          acc[i] += sum_{k=0..3} W[b,c,u,k] * x[c,u,k]
        //      and then widens the result from int16 → int32.
        //
        //      After processing Unroll vectors, a reduction tree sums the partial accumulators:
        //          acc[0] = sum_{u=0...Unroll-1} acc[u]
        //
        //  4. Add biases:
        //      The first register in the unroll loop is initialized with the bias:
        //          acc[0] = b[b * D32Lanes : (b+1) * D32Lanes]
        //      The remaining Unroll vectors start at zero and accumulate partial results.
        //
        //  5. Finally, store the output into a padded buffer and copy the first Rows_ elements
        //    into the output array.
        // ============================================================================
        template <Params<Kernels::SIMD> P>
            requires(std::is_same_v<scalar_t<P.types.in>, int8_t> && std::is_same_v<scalar_t<P.types.out>, int32_t>)
        struct Layer<Kernels::SIMD, P> {
            static constexpr auto params        = P;
            using input_type                    = scalar_t<params.types.in>;
            using output_type                   = scalar_t<params.types.out>;
            using extent_type                   = size_t;
            static constexpr extent_type Cols_  = params.dims.in;
            static constexpr extent_type Rows_  = params.dims.out;
            static constexpr extent_type Unroll = params.opt.unroll;
            static constexpr extent_type Pack   = sizeof(output_type) / sizeof(input_type);
            static constexpr extent_type Cols   = Cols_;

            static_assert(Cols % (Pack * Unroll) == 0);

            using Di8    = hn::ScalableTag<input_type>;
            using Du8    = hn::RebindToUnsigned<Di8>;
            using Di16   = hn::RepartitionToWide<Di8>;
            using Di32   = hn::RepartitionToWide<Di16>;
            using Veci8  = hn::VFromD<Di8>;
            using Vecu8  = hn::VFromD<Du8>;
            using Veci16 = hn::VFromD<Di16>;
            using Veci32 = hn::VFromD<Di32>;

            static_assert(std::is_same_v<hn::TFromD<Di32>, output_type>);

            static constexpr extent_type Chunks = Cols / (Pack * Unroll);

            // Lanes migh or might not be constexpr (HWY_CONSTEXPR expands to constexpr or nothing)
            // However, we can still take advantage of the lane size information when we have it at compile time,
            // by switching between dynamic / and static extents for mdspans/mdarray.
            // In case of a mdspan, all offset computations are done at compile time
            // However, we can not use stack storage for weights, which is fine because it is only allocated once
            static HWY_LANES_CONSTEXPR extent_type D8Lanes  = hn::Lanes(Di8());
            static HWY_LANES_CONSTEXPR extent_type D32Lanes = hn::Lanes(Di32());

            static HWY_LANES_CONSTEXPR extent_type Rows       = utils::pad_up(Rows_, D32Lanes);
            static HWY_LANES_CONSTEXPR extent_type RowPadding = Rows - Rows_;
            static HWY_LANES_CONSTEXPR extent_type Blocks     = Rows / D32Lanes;

            using weights_extent_t = extents<extent_type, nn::extent_if_constexpr_v<Blocks>, Chunks, Unroll,
                                             nn::extent_if_constexpr_v<D8Lanes>>;
            using biases_extent_t =
                extents<extent_type, nn::extent_if_constexpr_v<Blocks>, nn::extent_if_constexpr_v<D32Lanes>>;
            using packed_input_extent_t  = extents<extent_type, Chunks, Unroll>;
            using padded_output_extent_t = biases_extent_t;

            using packed_input_view_t = mdspan<const output_type, packed_input_extent_t>;
            using weights_array_t =
                mdarray<const input_type, weights_extent_t, layout_right, hwy::AlignedVector<input_type>>;
            using biases_array_t =
                mdarray<const output_type, biases_extent_t, layout_right, hwy::AlignedVector<output_type>>;
            using output_array_t =
                mdarray<output_type, padded_output_extent_t, layout_right, hwy::AlignedVector<output_type>>;

            static HWY_LANES_CONSTEXPR weights_extent_t       WeightsExtent{Blocks, Chunks, Unroll, D8Lanes};
            static HWY_LANES_CONSTEXPR biases_extent_t        BiasesExtent{Blocks, D32Lanes};
            static HWY_LANES_CONSTEXPR packed_input_extent_t  PackedInputExtent{Chunks, Unroll};
            static HWY_LANES_CONSTEXPR padded_output_extent_t PaddedOutputExtent = BiasesExtent;

            // for every target, both versions will be compiled, the best one will be chosen at runtime
            static constexpr auto dot = [](const Vecu8 a, const Veci8 b, const Veci32 acc) {
                if constexpr (params.opt.operation == decltype(params.opt)::Operation::SumOfMulQuadAcc) {
                    return hn::SumOfMulQuadAccumulate(Di32(), a, b, acc);
                } else if constexpr (params.opt.operation == decltype(params.opt)::Operation::SumOfMulPairAdd) {
                    // This version can overflow but is much faster that the exact emulation of the opeeration
                    const Veci16 sum0 = hn::SatWidenMulPairwiseAdd(Di16(), a, b);
                    const Veci32 sum1 = hn::WidenMulPairwiseAdd(Di32(), sum0, hn::Set(Di16(), 1));
                    return hn::Add(acc, sum1);
                } else {
                    static_assert(false, "No int8*int8->int32 operation was found");
                    return hn::Undefined(Di32());
                }
            };

            void load_weights(const input_type* HWY_RESTRICT w_ptr, const output_type* HWY_RESTRICT b_ptr) {
                m_weights = weights_array_t{WeightsExtent};
                m_biases  = biases_array_t{BiasesExtent};

                using namespace matrix;
                const auto transformed_weights =
                    tile_cols(tile_cols(pad(MatrixView{w_ptr, Rows_, Cols_}, RowPadding, 0), Pack), D8Lanes);
                const auto transformed_biases = pad(MatrixView{b_ptr, Rows_, 1}, RowPadding, 0);

                materialize(transformed_weights, m_weights.data());
                materialize(transformed_biases, m_biases.data());
            }

            template <extent_type N, typename GetFunc>
            static HWY_INLINE Veci32 reduce_tree(GetFunc&& get) {
                if constexpr (N == 1) {
                    return get(0);
                } else {
                    constexpr extent_type Half  = N / 2;
                    const Veci32          left  = reduce_tree<Half>([&](const size_t i) { return get(i); });
                    const Veci32          right = reduce_tree<N - Half>([&](const size_t i) { return get(i + Half); });
                    return left + right;
                }
            }

            void forward(const input_type* HWY_RESTRICT input_ptr, output_type* HWY_RESTRICT output_ptr) const {
                thread_local output_array_t tmp_out{PaddedOutputExtent};
                const packed_input_view_t   packed_input{reinterpret_cast<const output_type*>(input_ptr),
                                                       PackedInputExtent};

                for (extent_type b = 0; b < m_weights.extent(0); ++b) {
                    nn::RegisterBank<Unroll, Veci32>::run(
                        [&](const size_t u) { return u == 0 ? hn::Load(Di32(), &m_biases[b, 0]) : hn::Zero(Di32()); },
                        [&](auto get_reg, auto set_reg) {
                            for (extent_type c = 0; c < m_weights.extent(1); ++c) {
                                for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                                    set_reg(u, dot(hn::BitCast(Du8(), hn::Set(Di32(), packed_input[c, u])),
                                                   hn::Load(Di8(), &m_weights[b, c, u, 0]), get_reg(u)));
                                }
                                const Veci32 out = reduce_tree<Unroll>([&](size_t i) { return get_reg(i); });
                                hn::Store(out, Di32(), &tmp_out[b, 0]);
                            }
                        });
                    std::memcpy(output_ptr, tmp_out.data(), Rows_ * sizeof(output_type));
                }
            }

          private:
            weights_array_t m_weights;
            biases_array_t  m_biases;
        };
    }; // namespace HWY_NAMESPACE

} // namespace chepp::nnue::layers::affine

HWY_AFTER_NAMESPACE();

#endif