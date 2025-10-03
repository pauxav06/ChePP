//
// Created by paul on 9/26/25.
//

#ifndef CHEPP_LAYERS_H
#define CHEPP_LAYERS_H

#include "simd/operations.h"
#include "transformations.hpp"

#include <bits/local_lim.h>
#include <cstring>

constexpr int pad_up(const int n, const int m) noexcept
{
    return ((n + m - 1) / m) * m;
}

constexpr int pad_down(const int x, const int n) noexcept
{
    if (n == 0)
        return x;
    return (x >= 0) ? (x / n) * n : ((x - n + 1) / n) * n;
}



namespace affine
{
    enum Kernel
    {
        VecQuadRowMajor, // x86 VNNI / Neon DotProd / Neon >= 8 (fallback, slower) / x86 >= SSE3 (fallback, slower)
        Vec,             // x86 <= SSE2 / Neon <= 7
        Scalar,          // Other
    };

    template <size_t Rows_, size_t Columns_, Kernel kernel>
    struct AffineLayerImpl;

    template <size_t Rows_, size_t Columns_>
    struct AffineLayerImpl<Rows_, Columns_, VecQuadRowMajor>
    {

        static constexpr size_t Columns = Columns_;
        static_assert(Columns % 4 == 0); // we use the quad add op so we need at least 4 columns

        using D8  = simd::pick_tag_t<int8_t>;
        using D32 = simd::pick_tag_t<int32_t>;
        static_assert(simd::lane_count_v<D8> == simd::lane_count_v<D32> * 4); // same register width, should be true
        using Vec8  = simd::register_type_t<D8>;
        using Vec32 = simd::register_type_t<D32>;

        static_assert(simd::is_op_supported_v<simd::mul_quad_add<D8>, Vec8, Vec8, Vec32> ||
                          simd::is_op_supported_v<simd::mul_quad_add_fallback<D8>, Vec8, Vec8, Vec32>,
                      "Kernel requires mul quad add or fallback");

        static constexpr auto select_mul_quad_add()
        {
            if constexpr (!simd::is_op_supported_v<simd::mul_quad_add<D8>, Vec8, Vec8, Vec32>)
                return simd::mul_quad_add_fallback<D8>;
            else
                return simd::mul_quad_add<D8>;
        }

        static constexpr bool uses_fallback = !simd::is_op_supported_v<simd::mul_quad_add<D8>, Vec8, Vec8, Vec32>;
        static constexpr auto mul_quad_add  = select_mul_quad_add();

        static constexpr size_t Rows       = pad_up(Rows_, simd::lane_count_v<D32>);
        static constexpr size_t RowPadding = Rows - Rows_;

        static constexpr size_t TotBytes = Rows * Columns;
        static constexpr size_t TotRegs  = std::min((simd::register_count_v<simd::arch_t<D8>>) /
                                                        (2 + uses_fallback), // fallback uses a extra temp register
                                                    TotBytes / simd::lane_count_v<D8>);

        static constexpr size_t RegistersPerRow = Rows / simd::lane_count_v<D32>;
        static constexpr size_t Unroll          = std::max(1, pad_down(TotRegs / RegistersPerRow, 2));
        static constexpr size_t Chunks          = Columns / 4 / Unroll;

        void load_weights(const int8_t* weights, const int32_t* biases)
        {
            // these transformations ensure:
            // 1. That we never get UB on vectorised loads (padding)
            // 2. That we do the least amount of permutes during evaluation (tiling)
            AnyMatrixView<int8_t> weightsView{MatrixView{weights, Rows_, Columns_, Columns_}};
            weightsView = tile_cols(weightsView, 4);
            weightsView = pad(weightsView, RowPadding * 4, 0);
            weightsView.materialize(&m_weights[0][0][0][0]);
            AnyMatrixView<int32_t> biasesView{MatrixView{biases, Rows_, 1, 1}};
            biasesView = pad(biasesView, 0, RowPadding);
            biasesView.materialize(&m_biases[0][0]);
        }

        template<size_t Step, size_t Unroll>
        inline void tree_reduce(Vec32 v_acc[Unroll][RegistersPerRow]) const {
            if constexpr (Step < Unroll) {
                for (size_t r = 0; r < RegistersPerRow; ++r) {
                    for (size_t u = 0; u + Step < Unroll; u += 2 * Step) {
                        v_acc[u][r] = simd::add<D32>(v_acc[u][r], v_acc[u + Step][r]);
                    }
                }
                tree_reduce<Step * 2, Unroll>(v_acc);
            }
        }

        // the compiler sometimes thinks he needs to spill the accumulators costing about 30% performance
        // we prevent inlining as it makes it easier for the register allocator to understand everything
        // can happend in register
        void forward(const int8_t* CHEPP_RESTRICT input, int32_t* CHEPP_RESTRICT output) const
        {
            using namespace simd;
            Vec32 v_acc[Unroll][RegistersPerRow];

            // we have asserted the inputs are a multiple of 4, we should not read past
            const auto input_packed = reinterpret_cast<const int32_t*>(input);

            for (size_t u = 0; u < Unroll; ++u)
            {
                for (size_t r = 0; r < RegistersPerRow; ++r)
                {
                    // first unroll group gets biases and will be the one we need to read from at the end
                    if (u == 0)
                        v_acc[u][r] = load<D32>(m_biases[r]);
                    else
                        v_acc[u][r] = set1<D32>(0);
                }
            }

            for (size_t c = 0; c < Chunks; ++c)
            {
                for (size_t u = 0; u < Unroll; ++u)
                {
                    const Vec32 v_in_packed = set1<D32>(input_packed[c * Unroll + u]);
                    const Vec8  v_in        = *reinterpret_cast<const Vec8*>(&v_in_packed); // for neon since types are
                                                                                    // different
                    for (size_t r = 0; r < RegistersPerRow; ++r)
                    {
                        const Vec8 v_weights = load<D8>(m_weights[c][u][r]);
                        v_acc[u][r]          = simd::call_or_default<mul_quad_add, Vec32>(v_in, v_weights, v_acc[u][r]);
                    }
                }
            }

            tree_reduce<1, Unroll>(v_acc);
            alignas(64) int32_t tmp_out[RegistersPerRow * simd::lane_count_v<D32>];
            for (size_t r = 0; r < RegistersPerRow; ++r)
            {
                simd::store<D32>(&tmp_out[r * simd::lane_count_v<D32>], v_acc[0][r]);
            }

            for (size_t row = 0; row < Rows_; ++row)
            {
                output[row] = tmp_out[row];
            }
        }

        alignas(64) int8_t m_weights[Chunks][Unroll][RegistersPerRow][simd::lane_count_v<D8>] = {};
        alignas(64) int32_t m_biases[RegistersPerRow][simd::lane_count_v<D32>]                = {};

        static_assert(sizeof(m_weights) == Rows * Columns * sizeof(int8_t));
    };

    template <size_t Rows_, size_t Columns_>
    struct AffineLayerImpl<Rows_, Columns_, Scalar>
    {

        static constexpr size_t Rows    = Rows_;
        static constexpr size_t Columns = Columns_;

        void load_weights(const int8_t* weights, const int32_t* biases)
        {
            for (size_t r = 0; r < Rows; ++r)
            {
                for (size_t c = 0; c < Columns; ++c)
                {
                    m_weights[r][c] = weights[r * Columns + c];
                }
                m_biases[r] = biases[r];
            }
        }

        void forward(const int8_t* CHEPP_RESTRICT input, int32_t* CHEPP_RESTRICT output) const
        {
            for (size_t r = 0; r < Rows; ++r)
            {
                int32_t acc = m_biases[r];
                for (size_t c = 0; c < Columns; ++c)
                {
                    acc += static_cast<int32_t>(input[c]) * static_cast<int32_t>(m_weights[r][c]);
                }
                output[r] = acc;
            }
        }

      private:
        int8_t  m_weights[Rows][Columns] = {};
        int32_t m_biases[Rows]           = {};
    };

    template <size_t Rows_, size_t Columns_>
    struct AffineLayer
    {
        void load_weights(const int8_t* weights, const int32_t* biases) { m_impl.load_weights(weights, biases); }
        void forward(const int8_t* CHEPP_RESTRICT input, int32_t* CHEPP_RESTRICT output) const
        {
            m_impl.forward(input, output);
        }

        static constexpr Kernel m_kernel = []()
        {
            using D8    = simd::pick_tag_t<int8_t>;
            using D32   = simd::pick_tag_t<int32_t>;
            using Vec8  = simd::register_type_t<D8>;
            using Vec32 = simd::register_type_t<D32>;
            if constexpr (simd::is_op_supported_v<simd::mul_quad_add<D8>, Vec8, Vec8, Vec32> ||
                          simd::is_op_supported_v<simd::mul_quad_add_fallback<D8>, Vec8, Vec8, Vec32>)
                return VecQuadRowMajor;
            else
                return Scalar;
        }();
        AffineLayerImpl<Rows_, Columns_, m_kernel> m_impl{};
    };
} // namespace affine

namespace relu
{

    enum Kernel
    {
        Vec,
        Scalar
    };

    template <size_t N, size_t S, Kernel kernel>
    struct QuantizedClippedRelu16_8_Impl;

    template <size_t N, size_t S = 0>
    struct QuantizedClippedRelu16_8
    {
      private:
        static constexpr Kernel m_kernel = []() constexpr
        {
            using D16 = simd::pick_tag_t<int16_t, N / 2>;
            using D8  = simd::pick_tag_t<int8_t, N / 2>;
            if constexpr (simd::is_op_supported_v<simd::saturate_downcast2<D16>, simd::register_type_t<D16>,
                                                  simd::register_type_t<D16>, simd::register_type_t<D8>>)
                return Vec;
            else
                return Scalar;
        }();

      public:
        static void forward(const int16_t* CHEPP_RESTRICT in, int8_t* CHEPP_RESTRICT out)
        {
            QuantizedClippedRelu16_8_Impl<N, S, m_kernel>::forward(in, out);
        }
    };

    template <size_t N, size_t S>
    struct QuantizedClippedRelu16_8_Impl<N, S, Vec>
    {
        using D16   = simd::pick_tag_t<int16_t, N / 2>;
        using D8    = simd::pick_tag_t<int8_t, N / 2>;
        using Vec16 = simd::register_type_t<D16>;
        using Vec8  = simd::register_type_t<D8>;

        static void forward(const int16_t* __restrict in, int8_t* __restrict out)
            requires(N % simd::lane_count_v<D8> == 0)
        {
            using namespace simd;
            const auto v_in  = reinterpret_cast<const Vec16*>(in);
            auto       v_out = reinterpret_cast<Vec8*>(out);

            for (size_t i = 0; i < N / lane_count_v<D16>; i += 2)
            {
                v_out[i] = max<D8>(
                        simd::call_or_default<saturate_downcast2<D16>, Vec8>(
                            shr<D16>(v_in[i], S), shr<D16>(v_in[i + 1]), S),
                        set1<D8>(0));
            }
        }
    };

    template <size_t N, size_t S>
    struct QuantizedClippedRelu16_8_Impl<N, S, Scalar>
    {
        static void forward(const int16_t* CHEPP_RESTRICT in, int8_t* CHEPP_RESTRICT out)
        {
            for (size_t i = 0; i < N; ++i)
            {
                int16_t val = in[i] >> S;
                out[i]      = static_cast<int8_t>(val < 0 ? 0 : (val > 127 ? 127 : val));
            }
        }
    };

    template <size_t N, size_t S, Kernel kernel>
    struct ClippedRelu32_8_Impl;

    template <size_t N, size_t S>
    struct ClippedRelu32_8_Impl<N, S, Vec>
    {
        using D32   = simd::pick_tag_t<int32_t, N / 4>;
        using D16   = simd::pick_tag_t<int16_t, N / 4>;
        using D8    = simd::pick_tag_t<int8_t, N / 4>;
        using Vec32 = simd::register_type_t<D32>;
        using Vec16 = simd::register_type_t<D16>;
        using Vec8  = simd::register_type_t<D8>;

        static void forward(const int32_t* CHEPP_RESTRICT in, int8_t* CHEPP_RESTRICT out)
            requires(N % simd::lane_count_v<D8> == 0)
        {
            using namespace simd;
            const auto v_in  = reinterpret_cast<const Vec32*>(in);
            auto       v_out = reinterpret_cast<Vec8*>(out);

            for (size_t i = 0; i < N / simd::lane_count_v<D32>; i += 4)
            {
                Vec32       a   = shr<D32>(v_in[i], S);
                Vec32       b   = shr<D32>(v_in[i + 1], S);
                Vec32       c   = shr<D32>(v_in[i + 2], S);
                Vec32       d   = shr<D32>(v_in[i + 3], S);
                const Vec16 a16 = simd::call_or_default<saturate_downcast2<D32>, Vec16>(a, b);
                const Vec16 b16 = simd::call_or_default<saturate_downcast2<D32>, Vec16>(c, d);
                v_out[i] = max<D8>(
                        simd::call_or_default<saturate_downcast2<D16>, Vec8>(a16, b16),
                        set1<D8>(0));
            }
        }
    };

    template <size_t N, size_t S>
    struct ClippedRelu32_8_Impl<N, S, Scalar>
    {
        static void forward(const int32_t* CHEPP_RESTRICT in, int8_t* CHEPP_RESTRICT out)
        {
            for (size_t i = 0; i < N; ++i)
            {
                int32_t val = in[i] >> S;
                out[i]      = static_cast<int8_t>(val < 0 ? 0 : (val > 127 ? 127 : val));
            }
        }
    };

    template <size_t N, size_t S = 0>
    struct ClippedRelu32_8
    {
      private:
        static constexpr Kernel m_kernel = []() constexpr
        {
            using D32 = simd::pick_tag_t<int32_t, N / 4>;
            using D16 = simd::pick_tag_t<int16_t, N / 4>;
            using D8  = simd::pick_tag_t<int8_t, N / 4>;
            if constexpr (simd::is_op_supported_v<simd::saturate_downcast2<D32>, simd::register_type_t<D32>,
                                                  simd::register_type_t<D32>, simd::register_type_t<D16>>)
                return Vec;
            else
                return Scalar;
        }();

      public:
        static void forward(const int32_t* __restrict in, int8_t* __restrict out)
        {
            ClippedRelu32_8_Impl<N, S, m_kernel>::forward(in, out);
        }
    };

} // namespace relu

#endif // CHEPP_LAYERS_H
