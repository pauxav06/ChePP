#ifndef CHEPP_NNUE_UTILS_H_
#define CHEPP_NNUE_UTILS_H_

#include <chrono>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <thread>
#include <type_traits>
#include "core.h"

#include <hwy/aligned_allocator.h>

namespace chepp::nnue::utils {

    template<typename T>
    struct type_name
    {
        static constexpr std::string_view value = "unknown";
    };

    template<> struct type_name<std::int8_t>   { static constexpr std::string_view value = "int8_t"; };
    template<> struct type_name<std::int16_t>  { static constexpr std::string_view value = "int16_t"; };
    template<> struct type_name<std::int32_t>  { static constexpr std::string_view value = "int32_t"; };
    template<> struct type_name<std::int64_t>  { static constexpr std::string_view value = "int64_t"; };

    template<> struct type_name<std::uint8_t>  { static constexpr std::string_view value = "uint8_t"; };
    template<> struct type_name<std::uint16_t> { static constexpr std::string_view value = "uint16_t"; };
    template<> struct type_name<std::uint32_t> { static constexpr std::string_view value = "uint32_t"; };
    template<> struct type_name<std::uint64_t> { static constexpr std::string_view value = "uint64_t"; };

    template<> struct type_name<float>         { static constexpr std::string_view value = "float"; };
    template<> struct type_name<double>        { static constexpr std::string_view value = "double"; };
    template<> struct type_name<long double>   { static constexpr std::string_view value = "long double"; };

    template<typename T>
    inline constexpr std::string_view type_name_v = type_name<T>::value;

    template <typename T>
        requires std::is_integral_v<T>
    constexpr T
    pad_up(const T n, const T m) noexcept {
        return ((n + m - 1) / m) * m;
    }

    template <typename T>
        requires std::is_integral_v<T>
    constexpr T
    pad_down(const T x, const T n) noexcept {
        if (n == 0) return x;
        return (x >= 0) ? (x / n) * n : ((x - n + 1) / n) * n;
    }

    template <typename T>
        requires std::is_integral_v<T>
    constexpr bool
    is_power_of_two(const T x) {
        return x != 0 && (x & (x - 1)) == 0;
    }

    using extent_type = std::size_t;

    template <typename Container>
    void
    fill_random(Container&                     c,
                typename Container::value_type min  = std::numeric_limits<typename Container::value_type>::min(),
                typename Container::value_type max  = std::numeric_limits<typename Container::value_type>::max(),
                unsigned                       seed = 1234) {
        using T = Container::value_type;
        std::mt19937 rng(seed);

        using dist_t = std::conditional_t<std::is_integral_v<T>,
                                          std::conditional_t<std::is_signed_v<T>,
                                                             std::uniform_int_distribution<std::intmax_t>,
                                                             std::uniform_int_distribution<std::uintmax_t>>,
                                          std::uniform_real_distribution<long double>>;
        using res_t  = typename dist_t::result_type;

        dist_t dist(min, max);

        for (auto& x : c) {
            x = static_cast<T>(std::clamp(dist(rng), static_cast<res_t>(min), static_cast<res_t>(max)));
        }
    }

    template <int Begin, int End, int Step = 1, typename Func>
        requires((End - Begin) % Step == 0)
    HWY_INLINE void
    constexpr_for(Func&& f) {
        [&]<std::size_t... IS>(std::integer_sequence<std::size_t, IS...>) {
            (f(std::integral_constant<std::size_t, Begin + IS * Step>{}), ...);
        }(std::make_integer_sequence<std::size_t, (End - Begin) / Step>{});
    }

    struct BenchmarkResult {
        size_t iterations;
        std::size_t ns_per_iteration;

        std::size_t cost() const {
            return ns_per_iteration;
        }
    };

    template <typename TP, typename REP>
    inline BenchmarkResult benchmark(std::function<std::size_t()> func, std::chrono::duration<TP, REP> dur) {
        size_t iterations = 0;
        using namespace std::chrono_literals;
        std::jthread worker([&](std::stop_token st){
            std::size_t sum = 0;
            while (!st.stop_requested()) {
                sum += func();
                ++iterations;
            }
            hwy::PreventElision(sum);
        });

        std::this_thread::sleep_for(dur);

        worker.request_stop();

        std::size_t time_per_iter = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() / iterations;
        return BenchmarkResult{iterations, time_per_iter};
    }


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

#define REG_NAME(I) reg##I

#define DECL_REG_LOCAL_COND(I, V, ...) [[maybe_unused]] V REG_NAME(I);

#define REG_ARR_REF_COND(I, ...) &REG_NAME(I),

#define DECLARE_REG_BANK(N, V)                                                                                         \
    static_assert((N) <= 32, "Cannot declare more than 32 registers");                                                 \
    REPEAT(32, DECL_REG_LOCAL_COND, V)                                                                                 \
    std::array<V*, 32> regs_{REPEAT(32, REG_ARR_REF_COND)};                                                            \
    std::span<V*, N>   regs{regs_.data(), N};

#define GET_REG(N) *regs[N]

} // namespace chepp::nnue::utils

#endif
