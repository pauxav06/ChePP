#ifndef CHEPP_UTILS_H
#define CHEPP_UTILS_H

#include "expected.h"
#include "format.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string_view>
#include <type_traits>

#include <functional>
#include <hedley.h>

namespace chepp::utils {
    template <typename T>
    struct type_name;

    template <>
    struct type_name<std::int8_t> {
        static constexpr std::string_view value = "int8_t";
    };
    template <>
    struct type_name<std::int16_t> {
        static constexpr std::string_view value = "int16_t";
    };
    template <>
    struct type_name<std::int32_t> {
        static constexpr std::string_view value = "int32_t";
    };
    template <>
    struct type_name<std::int64_t> {
        static constexpr std::string_view value = "int64_t";
    };

    template <>
    struct type_name<std::uint8_t> {
        static constexpr std::string_view value = "uint8_t";
    };
    template <>
    struct type_name<std::uint16_t> {
        static constexpr std::string_view value = "uint16_t";
    };
    template <>
    struct type_name<std::uint32_t> {
        static constexpr std::string_view value = "uint32_t";
    };
    template <>
    struct type_name<std::uint64_t> {
        static constexpr std::string_view value = "uint64_t";
    };

    template <>
    struct type_name<float> {
        static constexpr std::string_view value = "float";
    };
    template <>
    struct type_name<double> {
        static constexpr std::string_view value = "double";
    };
    template <>
    struct type_name<long double> {
        static constexpr std::string_view value = "long double";
    };
    template <typename T>
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

    // TODO fix this with a safe function
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
    HEDLEY_ALWAYS_INLINE void
    constexpr_for(Func&& f) {
        [&]<std::size_t... IS>(std::integer_sequence<std::size_t, IS...>) {
            (f(std::integral_constant<std::size_t, Begin + IS * Step>{}), ...);
        }(std::make_integer_sequence<std::size_t, (End - Begin) / Step>{});
    }

    template <typename Tuple, typename F, std::size_t... I>
    constexpr void
    tuple_for_each_impl(Tuple&& t, F&& f, std::index_sequence<I...>) {
        (f(std::integral_constant<std::size_t, I>{}, std::get<I>(t)), ...);
    }

    template <typename Tuple, typename F>
    constexpr void
    tuple_for_each(Tuple&& t, F&& f) {
        auto idx_sq = std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{};
        tuple_for_each_impl(std::forward<Tuple>(t), std::forward<F>(f), idx_sq);
    }

    template <std::size_t... Is, typename F>
    constexpr auto
    make_tuple_from_sequence(F&& f, std::index_sequence<Is...>) {
        return std::tuple{f(std::integral_constant<std::size_t, Is>{})...};
    }

    template <std::size_t N, typename F>
    constexpr auto
    make_tuple(F&& f) {
        return make_tuple_from_sequence(std::forward<F>(f), std::make_index_sequence<N>{});
    }

    template <std::size_t... Is, typename F>
    constexpr auto
    make_array_from_sequence(F&& f, std::index_sequence<Is...>) {
        using T = decltype(f(size_t{0}));
        return std::array<T, sizeof...(Is)>{f(Is)...};
    }

    template <std::size_t N, typename F>
    constexpr auto
    make_array(F&& f) {
        return make_array_from_sequence(std::forward<F>(f), std::make_index_sequence<N>{});
    }
} // namespace chepp::utils

#if (defined __CDT_PARSER__) || (defined __INTELLISENSE__) || (defined Q_CREATOR_RUN) || (defined __CLANGD__) ||       \
    (defined GROK_ELLIPSIS_BUILD) || (defined __JETBRAINS_IDE__)
#define IDE 1
#else
#define IDE 0
#endif

#endif // CHEPP_HT_H
