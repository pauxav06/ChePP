#pragma once

#include <algorithm>

#include "traits.h"

namespace simd
{
    // we use false type to detect a non-available op.
    // be careful keeping it in a constexpr if / else chain
    // otherwise the compiler will not be able to deduce the output type correctly

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    add(
        const register_type_t<Vec> a,
        const register_type_t<Vec> b) noexcept
    {
        using Arch = arch_t<Vec>;
        using T    = value_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            return a + b;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm_add_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_add_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_add_epi32(a, b);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm256_add_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_add_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_add_epi32(a, b);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm512_add_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_add_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_add_epi32(a, b);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vaddq_s8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return vaddq_s16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return vaddq_s32(a, b);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    sub(
        const register_type_t<Vec> a,
        const register_type_t<Vec> b) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            return a - b;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm_sub_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_sub_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_sub_epi32(a, b);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm256_sub_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_sub_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_sub_epi32(a, b);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm512_subs_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_subs_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_subs_epi32(a, b);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vsubq_s8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return vsubq_s16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return vsubq_s32(a, b);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    loadu(const value_t<Vec>* CHEPP_RESTRICT ptr) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            return *ptr;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vld1q_s8(reinterpret_cast<const int8_t*>(ptr));
            if constexpr (std::is_same_v<T, int16_t>)
                return vld1q_s16(reinterpret_cast<const int16_t*>(ptr));
            if constexpr (std::is_same_v<T, int32_t>)
                return vld1q_s32(reinterpret_cast<const int32_t*>(ptr));
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    storeu(
        value_t<Vec>* CHEPP_RESTRICT ptr,
        const register_type_t<Vec> val) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            *ptr = val;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), val);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), val);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                vst1q_s8(reinterpret_cast<int8_t*>(ptr), val);
            if constexpr (std::is_same_v<T, int16_t>)
                vst1q_s16(reinterpret_cast<int16_t*>(ptr), val);
            if constexpr (std::is_same_v<T, int32_t>)
                vst1q_s32(reinterpret_cast<int32_t*>(ptr), val);
        }
#endif
        else
            return UnsupportedSentinel{};
    };

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    load(const value_t<Vec>* CHEPP_RESTRICT ptr) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            return *ptr;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            return _mm_load_si128(ptr);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            return _mm512_load_si512(reinterpret_cast<const __m512i*>(ptr));
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vld1q_s8(reinterpret_cast<const int8_t*>(ptr));
            if constexpr (std::is_same_v<T, int16_t>)
                return vld1q_s16(reinterpret_cast<const int16_t*>(ptr));
            if constexpr (std::is_same_v<T, int32_t>)
                return vld1q_s32(reinterpret_cast<const int32_t*>(ptr));
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    store(
        value_t<Vec>* CHEPP_RESTRICT ptr,
        const register_type_t<Vec> val) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            *ptr = val;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(ptr), val);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), val);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), val);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                vst1q_s8(reinterpret_cast<int8_t*>(ptr), val);
            if constexpr (std::is_same_v<T, int16_t>)
                vst1q_s16(reinterpret_cast<int16_t*>(ptr), val);
            if constexpr (std::is_same_v<T, int32_t>)
                vst1q_s32(reinterpret_cast<int32_t*>(ptr), val);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    set1(const value_t<Vec> v) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            return v;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm_set1_epi8(v);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_set1_epi16(v);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_set1_epi32(v);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm256_set1_epi8(v);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_set1_epi16(v);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_set1_epi32(v);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm512_set1_epi8(v);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_set1_epi16(v);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_set1_epi32(v);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vdupq_n_s8(v);
            if constexpr (std::is_same_v<T, int16_t>)
                return vdupq_n_s16(v);
            if constexpr (std::is_same_v<T, int32_t>)
                return vdupq_n_s32(v);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    min(
        const register_type_t<Vec> a,
        const register_type_t<Vec> b) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
            return a < b ? a : b;
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm_min_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_min_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_min_epi32(a, b);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm256_min_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_min_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_min_epi32(a, b);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm512_min_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_min_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_min_epi32(a, b);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vminq_s8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return vminq_s16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return vminq_s32(a, b);
        }
#endif
        else
            return UnsupportedSentinel{};
    };

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    max(
        const register_type_t<Vec> a,
        const register_type_t<Vec> b) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
            return a > b ? a : b;
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm_max_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_max_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_max_epi32(a, b);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm256_max_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_max_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_max_epi32(a, b);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return _mm512_max_epi8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_max_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_max_epi32(a, b);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vmaxq_s8(a, b);
            if constexpr (std::is_same_v<T, int16_t>)
                return vmaxq_s16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return vmaxq_s32(a, b);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    mul_lo(
        const register_type_t<Vec> a,
        const register_type_t<Vec> b) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
            return a * b;
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_mullo_epi16(a, b);
#if defined(__SSE4_1__)
            else if constexpr (std::is_same_v<T, int32_t>)
                return _mm_mullo_epi32(a, b);
#endif
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_mullo_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_mullo_epi32(a, b);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_mullo_epi16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_mullo_epi32(a, b);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return vmulq_s16(a, b);
            if constexpr (std::is_same_v<T, int32_t>)
                return vmulq_s32(a, b);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto
    mul_hi(
        const register_type_t<Vec> a,
        const register_type_t<Vec> b) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            return static_cast<T>((static_cast<int32_t>(a) * static_cast<int32_t>(b)) >> 16);
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            return _mm_mulhi_epi16(a, b);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            return _mm256_mulhi_epi16(a, b);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            return _mm512_mulhi_epi16(a, b);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            return vmulhq_s16(a, b);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Tag>
    CHEPP_ALWAYS_INLINE auto
    mul_quad_add(
        const register_type_t<Tag> a,
        const register_type_t<Tag> b,
        const register_type_t<transform_type_t<Tag, int32_t>> acc) noexcept
        requires(std::is_same_v<value_t<Tag>, int8_t>)
    {

        if constexpr (false)
        {
        }
#if CHEPP_AVX512VNNI
        else if constexpr (bit_count_v<Tag> == 512)
        {
            return _mm512_dpbusds_epi32(acc, a, b);
        }
#endif
#if CHEPP_AVXVNNI
        else if constexpr (bit_count_v<Tag> == 256)
            return _mm256_dpbusds_avx_epi32(acc, a, b);
        else if constexpr (bit_count_v<Tag> == 128)
            return _mm_dpbusds_avx_epi32(acc, a, b);
#elif CHEPP_AVX512VNNI
        else if constexpr (bit_count_v<Tag> == 256)
            return _mm256_mask_dpbusds_epi32(acc, 0xFF, a, b);
        else if constexpr (bit_count_v<Tag> == 128)
            return _mm_mask_dpbusds_epi32(acc, 0xF, a, b);
#endif
#if CHEPP_NEON_DOT
        else if constexpr (bit_count_v<Tag> == 128)
            return vdotq_s32(acc, a, b);
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Tag>
    CHEPP_ALWAYS_INLINE auto
    mul_quad_add_fallback(
        const register_type_t<Tag> a,
        const register_type_t<Tag> b,
        const register_type_t<transform_type_t<Tag, int32_t>> acc) noexcept
        requires(std::is_same_v<value_t<Tag>, int8_t>)
    {
        if constexpr (false)
        {
        }
#if CHEPP_AVX512 && CHEPP_AVX2
        else if constexpr (bit_count_v<Tag> == 512)
        {
            __m256i lo_a   = _mm512_castsi512_si256(a);
            __m256i hi_a   = _mm512_extracti64x4_epi64(a, 1);
            __m256i lo_b   = _mm512_castsi512_si256(b);
            __m256i hi_b   = _mm512_extracti64x4_epi64(b, 1);
            __m256i lo_acc = _mm512_castsi512_si256(acc);
            __m256i hi_acc = _mm512_extracti64x4_epi64(acc, 1);

            lo_acc =
                _mm256_add_epi32(lo_acc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo_a, lo_b), _mm256_set1_epi16(1)));
            hi_acc =
                _mm256_add_epi32(hi_acc, _mm256_madd_epi16(_mm256_maddubs_epi16(hi_a, hi_b), _mm256_set1_epi16(1)));

            return _mm512_inserti64x4(_mm512_castsi256_si512(lo_acc), hi_acc, 1);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (bit_count_v<Tag> == 256)
        {
            __m256i tmp = _mm256_maddubs_epi16(a, b);
            tmp         = _mm256_madd_epi16(tmp, _mm256_set1_epi16(1));
            return _mm256_add_epi32(acc, tmp);
        }
#endif
#if CHEPP_SSE3
        else if constexpr (bit_count_v<Tag> == 128)
        {
            __m128i tmp = _mm_maddubs_epi16(a, b);
            tmp         = _mm_madd_epi16(tmp, _mm_set1_epi16(1));
            return _mm_add_epi32(acc, tmp);
        }
#endif
#if CHEPP_NEON_V8
        else if constexpr (bit_count_v<Tag> == 128)
        {
            int16x8_t mul_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
            int16x8_t mul_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));

            int32x4_t sum_lo = vpaddlq_s16(mul_lo);
            int32x4_t sum_hi = vpaddlq_s16(mul_hi);

            return vaddq_s32(acc, vaddq_s32(sum_lo, sum_hi));
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Tag>
    CHEPP_ALWAYS_INLINE auto
    saturate_downcast(const register_type_t<Tag> src) noexcept
        requires(std::is_same_v<value_t<Tag>, int32_t> || std::is_same_v<value_t<Tag>, int16_t>)
    {
        using T    = value_t<Tag>;
        using Arch = arch_t<T>;

        if constexpr (false)
        {
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Tag> == 128)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_packs_epi32(src, src);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_packs_epi16(src, src);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Tag> == 256)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_packs_epi32(src, src);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_packs_epi16(src, src);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Tag> == 512)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_cvtepi32_epi16(src);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_cvtepi16_epi8(src);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return vqmovn_s32(src);
            if constexpr (std::is_same_v<T, int16_t>)
                return vqmovn_s16(src);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Tag>
    CHEPP_ALWAYS_INLINE auto saturate_downcast2(
        const register_type_t<Tag> lo,
        const register_type_t<Tag> hi) noexcept
        requires(std::is_same_v<value_t<Tag>, int32_t> || std::is_same_v<value_t<Tag>, int16_t>)
    {
        using T    = value_t<Tag>;
        using Arch = arch_t<Tag>;

        if constexpr (false)
        {
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Tag> == 128)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_packs_epi32(lo, hi);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_packs_epi16(lo, hi);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Tag> == 256)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_packs_epi32(lo, hi);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_packs_epi16(lo, hi);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Tag> == 512)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_packs_epi32(lo, hi);
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_packs_epi16(lo, hi);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int32_t>)
                return vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
            if constexpr (std::is_same_v<T, int16_t>)
                return vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi));
        }
#endif
        else
            return UnsupportedSentinel{};
    }

    template <typename Vec>
    CHEPP_ALWAYS_INLINE auto shr(
        const register_type_t<Vec> a,
        const int count) noexcept
    {
        using T    = value_t<Vec>;
        using Arch = arch_t<Vec>;

        if constexpr (false)
        {
        }
        else if constexpr (std::is_same_v<Arch, Scalar>)
        {
            return a >> count;
        }
#if CHEPP_SSE2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 128)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm_srai_epi16(a, count);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm_srai_epi32(a, count);
        }
#endif
#if CHEPP_AVX2
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 256)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm256_srai_epi16(a, count);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm256_srai_epi32(a, count);
        }
#endif
#if CHEPP_AVX512
        else if constexpr (is_x86_v<Arch> && bit_count_v<Vec> == 512)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return _mm512_srai_epi16(a, count);
            if constexpr (std::is_same_v<T, int32_t>)
                return _mm512_srai_epi32(a, count);
        }
#endif
#if CHEPP_NEON
        else if constexpr (is_neon_v<Arch>)
        {
            if constexpr (std::is_same_v<T, int16_t>)
                return vshrq_n_s16(a, count);
            if constexpr (std::is_same_v<T, int32_t>)
                return vshrq_n_s32(a, count);
        }
#endif
        else
            return UnsupportedSentinel{};
    }

} // namespace simd
