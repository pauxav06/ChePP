#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "generated/cpu_features.h"

#if CHEPP_SSE2 || CHEPP_SSE3
#define CHEPP_SIMD_X86
#include <emmintrin.h>
#endif
#if CHEPP_AVX2 || CHEPP_AVX512
#define CHEPP_SIMD_X86
#include <immintrin.h>
#endif
#if CHEPP_NEON__always_inline
#define CHEPP_SIMD_NEON
#include <arm_neon.h>
#endif

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#if defined(_MSC_VER)
#  define CHEPP_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define CHEPP_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#  define CHEPP_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER)
#  define CHEPP_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#  define CHEPP_RESTRICT __restrict__
#else
#  define CHEPP_RESTRICT
#endif


namespace simd {
    struct ArchTag {
    };

    struct Scalar : ArchTag {
    };

    struct X86 : ArchTag {
    };

    struct SSE2 : X86 {
    };

    struct SSE3 : X86 {
    };

    struct AVX2 : X86 {
    };

    struct AVX512 : X86 {
    };

    struct Neon : ArchTag {
    };

    struct Neon32 : Neon {
    };

    struct Neon64 : Neon {
    };

    using selected_arch =
#if CHEPP_AVX512
    AVX512;
#elif CHEPP_AVX2
    AVX2;
#elif CHEPP_SSE3
        SSE3;
#elif CHEPP_SSE2
    SSE2;
#elif CHEPP_NEON_V8
    Neon64;
#elif CHEPP_NEON
    Neon32;
#else
    Scalar;
#endif


    template<typename T, typename Arch, std::size_t Bits>
    struct VecTag {
        using value_type = T;
        using arch = Arch;
        static constexpr std::size_t width_bits = Bits;
    };

    template<typename T>
    using VecScalar = VecTag<T, Scalar, sizeof(T) * 8>;
    template<typename T, typename Arch>
    using Vec128 = VecTag<T, Arch, 128>;
    template<typename T, typename Arch>
    using Vec256 = VecTag<T, Arch, 256>;
    template<typename T, typename Arch>
    using Vec512 = VecTag<T, Arch, 512>;


    template<typename Vec>
    struct bit_count;

    template<typename T, typename Arch, std::size_t Bits>
    struct bit_count<VecTag<T, Arch, Bits>> {
        static constexpr std::size_t value = Bits;
    };

    template<typename Vec>
    inline constexpr std::size_t bit_count_v = bit_count<Vec>::value;

    template <typename Tag>
    using arch_t = typename Tag::arch;

    template <typename Tag>
    using value_t = typename Tag::value_type;

    template<typename Vec, typename NewT>
    struct transform_type {
        static constexpr std::size_t width_bits =
            std::is_same_v<typename Vec::arch, Scalar> ? sizeof(NewT) * 8 : Vec::width_bits;

        using type = VecTag<NewT, typename Vec::arch, width_bits>;
    };

    template<typename Vec, typename NewT>
    using transform_type_t = typename transform_type<Vec, NewT>::type;

    template<typename Vec>
    struct lane_count;

    template<typename T, typename Arch, std::size_t Bits>
    struct lane_count<VecTag<T, Arch, Bits> > {
        static constexpr std::size_t value = Bits / (8 * sizeof(T));
    };

    template<typename Vec>
    inline constexpr std::size_t lane_count_v = lane_count<Vec>::value;


    template<typename Arch>
    struct register_count;

    template<>
    struct register_count<Scalar> {
        static constexpr std::size_t value = 32;
    };

    template<>
    struct register_count<SSE2> {
        static constexpr std::size_t value = 16;
    };

    template<>
    struct register_count<SSE3> {
        static constexpr std::size_t value = 16;
    };

    template<>
    struct register_count<AVX2> {
        static constexpr std::size_t value = 16;
    };

    template<>
    struct register_count<AVX512> {
        static constexpr std::size_t value = 32;
    };

    template<>
    struct register_count<Neon32> {
        static constexpr std::size_t value = 16;
    };

    template<>
    struct register_count<Neon64> {
        static constexpr std::size_t value = 32;
    };

    template<typename Arch>
    inline constexpr std::size_t register_count_v = register_count<Arch>::value;

    template<typename Arch>
    concept is_x86_v = std::is_base_of_v<X86, Arch>;

    template<typename Arch>
    concept is_neon_v = std::is_base_of_v<Neon, Arch>;

    template<typename VecTag>
    struct register_type;

    template<typename T>
    struct register_type<VecScalar<T> > {
        using type = T;
    };

#if CHEPP_SSE2 || CHEPP_SSE3
    template<typename T, typename Arch>
    requires (std::is_same_v<Arch, SSE2> || std::is_same_v<Arch, SSE3>)
    struct register_type<Vec128<T, Arch>> {
        using type = __m128i;
    };
#endif
#if CHEPP_AVX2
    template<typename T>
    struct register_type<Vec128<T, AVX2> > {
        using type = __m128i;
    };

    template<typename T>
    struct register_type<Vec256<T, AVX2> > {
        using type = __m256i;
    };
#endif
#if CHEPP_AVX512
    template<typename T>
    struct register_type<Vec128<T, AVX512> > {
        using type = __m128i;
    };

    template<typename T>
    struct register_type<Vec256<T, AVX512> > {
        using type = __m256i;
    };

    template<typename T>
    struct register_type<Vec512<T, AVX512> > {
        using type = __m512i;
    };
#endif

#if CHEPP_NEON
    template<>
    struct register_type<Vec128<int8_t, Neon32> > {
        using type = int8x16_t;
    };
    template<>
    struct register_type<Vec128<int16_t, Neon32> > {
        using type = int16x8_t;
    };
    template<>
    struct register_type<Vec128<int32_t, Neon32> > {
        using type = int32x4_t;
    };

    template<>
    struct register_type<Vec128<int8_t, Neon64> > {
        using type = int8x16_t;
    };
    template<>
    struct register_type<Vec128<int16_t, Neon64> > {
        using type = int16x8_t;
    };
    template<>
    struct register_type<Vec128<int32_t, Neon64> > {
        using type = int32x4_t;
    };
#endif

    template<typename Vec>
    using register_type_t = typename register_type<Vec>::type;

    struct UnsupportedSentinel {
        explicit UnsupportedSentinel() = default;
    };

    template <auto F, typename... Args>
    concept is_op_supported_v =
    requires(Args... args) {
        { F(args...) } -> std::same_as<decltype(F(args...))>;
    }
    && !std::is_same_v<decltype(F(std::declval<Args>()...)), UnsupportedSentinel>;


    // utility for cross compilation.
    // It will return either the result if to operation is defined,
    // either a default value of type RetT
    // It can be used as a placeholder to not get compile errors in case an operation
    // is not defined
    // It is the user responsability the path containing this code is never executed
    // if the operation is not defined
    // Availability can be checked at compile time with simd::is_op_supported_v<Tag>
    template <auto Op, typename RetT, typename... CallArgs>
    CHEPP_ALWAYS_INLINE RetT
    call_or_default(CallArgs&&... args) {
        if constexpr (is_op_supported_v<Op, std::remove_cvref_t<CallArgs>...>) {
            return Op(std::forward<CallArgs>(args)...);
        } else {
            return RetT{};
        }
    }

    template<typename T, std::size_t N = 0, typename Arch = selected_arch>
    struct pick_tag {
        using type = std::conditional_t<
            is_x86_v<Arch>,
            std::conditional_t<
                N == 0,
                std::conditional_t<CHEPP_AVX512, Vec512<T, Arch>,
                    std::conditional_t<CHEPP_AVX2, Vec256<T, Arch>,
                        std::conditional_t<CHEPP_SSE2 || CHEPP_SSE3, Vec128<T, Arch>,
                            VecScalar<T> > > >,
                std::conditional_t<N >= 512 && CHEPP_AVX512, Vec512<T, Arch>,
                    std::conditional_t<N >= 256 && CHEPP_AVX2, Vec256<T, Arch>,
                        std::conditional_t<N >= 128, Vec128<T, Arch>,
                            VecScalar<T> > > >
            >,
            std::conditional_t<is_neon_v<Arch>,
                Vec128<T, Arch>,
                VecScalar<T>
            >
        >;
    };

    template<typename T, std::size_t N = 0>
    using pick_tag_t = typename pick_tag<T, N>::type;
} // namespace simd
