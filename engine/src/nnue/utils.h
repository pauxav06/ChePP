#ifndef CHEPP_NNUE_UTILS_H_
#define CHEPP_NNUE_UTILS_H_

#include "core.h"
#include <chrono>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <thread>
#include <type_traits>

#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a##b

#define REPEAT_1(M, ...) M(0, __VA_ARGS__)
#define REPEAT_2(M, ...) REPEAT_1(M, __VA_ARGS__) M(1, __VA_ARGS__)
#define REPEAT_3(M, ...) REPEAT_2(M, __VA_ARGS__) M(2, __VA_ARGS__)
#define REPEAT_4(M, ...) REPEAT_3(M, __VA_ARGS__) M(3, __VA_ARGS__)
#define REPEAT_5(M, ...) REPEAT_4(M, __VA_ARGS__) M(4, __VA_ARGS__)
#define REPEAT_6(M, ...) REPEAT_5(M, __VA_ARGS__) M(5, __VA_ARGS__)
#define REPEAT_7(M, ...) REPEAT_6(M, __VA_ARGS__) M(6, __VA_ARGS__)
#define REPEAT_8(M, ...) REPEAT_7(M, __VA_ARGS__) M(7, __VA_ARGS__)
#define REPEAT_9(M, ...) REPEAT_8(M, __VA_ARGS__) M(8, __VA_ARGS__)
#define REPEAT_10(M, ...) REPEAT_9(M, __VA_ARGS__) M(9, __VA_ARGS__)
#define REPEAT_11(M, ...) REPEAT_10(M, __VA_ARGS__) M(10, __VA_ARGS__)
#define REPEAT_12(M, ...) REPEAT_11(M, __VA_ARGS__) M(11, __VA_ARGS__)
#define REPEAT_13(M, ...) REPEAT_12(M, __VA_ARGS__) M(12, __VA_ARGS__)
#define REPEAT_14(M, ...) REPEAT_13(M, __VA_ARGS__) M(13, __VA_ARGS__)
#define REPEAT_15(M, ...) REPEAT_14(M, __VA_ARGS__) M(14, __VA_ARGS__)
#define REPEAT_16(M, ...) REPEAT_15(M, __VA_ARGS__) M(15, __VA_ARGS__)
#define REPEAT_17(M, ...) REPEAT_16(M, __VA_ARGS__) M(16, __VA_ARGS__)
#define REPEAT_18(M, ...) REPEAT_17(M, __VA_ARGS__) M(17, __VA_ARGS__)
#define REPEAT_19(M, ...) REPEAT_18(M, __VA_ARGS__) M(18, __VA_ARGS__)
#define REPEAT_20(M, ...) REPEAT_19(M, __VA_ARGS__) M(19, __VA_ARGS__)
#define REPEAT_21(M, ...) REPEAT_20(M, __VA_ARGS__) M(20, __VA_ARGS__)
#define REPEAT_22(M, ...) REPEAT_21(M, __VA_ARGS__) M(21, __VA_ARGS__)
#define REPEAT_23(M, ...) REPEAT_22(M, __VA_ARGS__) M(22, __VA_ARGS__)
#define REPEAT_24(M, ...) REPEAT_23(M, __VA_ARGS__) M(23, __VA_ARGS__)
#define REPEAT_25(M, ...) REPEAT_24(M, __VA_ARGS__) M(24, __VA_ARGS__)
#define REPEAT_26(M, ...) REPEAT_25(M, __VA_ARGS__) M(25, __VA_ARGS__)
#define REPEAT_27(M, ...) REPEAT_26(M, __VA_ARGS__) M(26, __VA_ARGS__)
#define REPEAT_28(M, ...) REPEAT_27(M, __VA_ARGS__) M(27, __VA_ARGS__)
#define REPEAT_29(M, ...) REPEAT_28(M, __VA_ARGS__) M(28, __VA_ARGS__)
#define REPEAT_30(M, ...) REPEAT_29(M, __VA_ARGS__) M(29, __VA_ARGS__)
#define REPEAT_31(M, ...) REPEAT_30(M, __VA_ARGS__) M(30, __VA_ARGS__)
#define REPEAT_32(M, ...) REPEAT_31(M, __VA_ARGS__) M(31, __VA_ARGS__)

#define REPEAT(N, M, ...) CAT(REPEAT_, N)(M, __VA_ARGS__)

#define CALL_IMPL(N, V, ...) V
#define CALL_N(N, F) REPEAT(N, CALL_IMPL, F)

#define IF_ELSE(cond, val, other) _IF_ELSE(cond, val, other)
#define _IF_ELSE(cond, val, other) IF_##cond(val, other)

#define IF_1(val, other) val
#define IF_0(val, other) other

#define EXTENT_IF(cond, val) (IF_ELSE(cond, val, std::dynamic_extent))
#define HWY_CONSTEXPR_EXT(val) EXTENT_IF(HWY_HAVE_CONSTEXPR_LANES, val)
#define HWY_STATIC_CONSTEXPR IF_ELSE(HWY_HAVE_CONSTEXPR_LANES, static constexpr, )
#define HWY_CONST IF_ELSE(HWY_HAVE_CONSTEXPR_LANES, ,const)



#define REG_NAME(I) reg##I

#define DECL_REG_LOCAL_COND(I, V, ...) [[maybe_unused]] V REG_NAME(I);

#define REG_ARR_REF_COND(I, ...) &REG_NAME(I),

#define DECLARE_REG_BANK(N, V)                                                                                         \
    static_assert((N) <= 32, "Cannot declare more than 32 registers");                                                 \
    REPEAT(32, DECL_REG_LOCAL_COND, V)                                                                                 \
    std::array<V*, 32> regs_{REPEAT(32, REG_ARR_REF_COND)};                                                            \
    std::span<V*, N>   regs{regs_.data(), N};

#define GET_REG(N) *regs[N]

template <typename IT>
concept byte_input_or_output_iterator =
    requires { std::is_convertible_v<std::decay_t<typename std::iterator_traits<IT>::value_type>, uint8_t>; };

namespace chepp::nnue::utils {
    template <typename T, byte_input_or_output_iterator It>
    bool
    read_n(It& begin, const It& end, const std::size_t n) {
        return std::ranges::advance(begin, n * sizeof(T), end) == 0;
    }
} // namespace chepp::nnue::utils

#endif
