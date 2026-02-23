#ifndef CHEPP_UTILS_H
#define CHEPP_UTILS_H

#include <algorithm>
#include <cstdint>
#include <random>
#include <string_view>
#include <type_traits>

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
    struct type_suffix;

    template <>
    struct type_suffix<std::uint8_t> {
        static constexpr std::string_view value = "";
    };
    template <>
    struct type_suffix<std::uint16_t> {
        static constexpr std::string_view value = "";
    };
    template <>
    struct type_suffix<std::uint32_t> {
        static constexpr std::string_view value = "U";
    };
    template <>
    struct type_suffix<std::uint64_t> {
        static constexpr std::string_view value = "ULL";
    };
    template <>
    struct type_suffix<float> {
        static constexpr std::string_view value = "F";
    };
    template <>
    struct type_suffix<long double> {
        static constexpr std::string_view value = "L";
    };

    template <>
    struct type_suffix<std::int8_t> {
        static constexpr std::string_view value = "";
    };
    template <>
    struct type_suffix<std::int16_t> {
        static constexpr std::string_view value = "";
    };
    template <>
    struct type_suffix<std::int32_t> {
        static constexpr std::string_view value = "";
    };
    template <>
    struct type_suffix<std::int64_t> {
        static constexpr std::string_view value = "LL";
    };

    template <typename T>
    inline constexpr std::string_view type_name_v = type_name<T>::value;

    template <typename T>
    inline constexpr std::string_view type_suffix_v = type_suffix<T>::value;

    template <typename T>
        requires requires { type_suffix_v<T>; }
    std::string
    write_with_suffix(const T& value) {
        return std::string(value).append(type_suffix_v<T>);
    }

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

} // namespace chepp::utils

#endif // CHEPP_HT_H
