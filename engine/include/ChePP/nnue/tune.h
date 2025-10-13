#ifndef CHEPP_TUNE_H_
#define CHEPP_TUNE_H_

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace chepp::nnue::tune {

    // Small utility for generating cartesian products of lists of parameters for a grid search
    // Since we template our kernels to these params to generate all variations, and perform the search at runtime,
    // we need to use a bit of constexpr magic!

    template <size_t... Ns>
    consteval size_t prod_constexpr() {
        size_t r = 1;
        ((r *= Ns), ...);
        return r;
    }

    template <typename T>
    using array_value_t = T::value_type;

    template <typename T>
    inline constexpr size_t array_size_v = std::tuple_size_v<std::remove_reference_t<T>>;

    template <size_t K>
    consteval std::array<size_t, K> make_divisors_from_sizes(const std::array<size_t, K>& sizes) {
        std::array<size_t, K> divisors{};
        size_t                acc = 1;
        for (size_t i = 0; i < K; ++i) {
            divisors[K - 1 - i] = acc;
            acc *= sizes[K - 1 - i];
        }
        return divisors;
    }

    template <typename Dest, typename ArraysTuple, size_t K, size_t... Is>
    consteval Dest make_dest_for_index_impl(const ArraysTuple& arrays, const std::array<size_t, K>& divisors,
                                            const std::array<size_t, K>& sizes, size_t idx,
                                            std::index_sequence<Is...>) {
        return Dest{std::get<Is>(arrays)[(idx / divisors[Is]) % sizes[Is]]...};
    }

    template <typename Dest, typename ArraysTuple, size_t K>
    consteval Dest make_dest_for_index(const ArraysTuple& arrays, const std::array<size_t, K>& divisors,
                                       const std::array<size_t, K>& sizes, size_t idx) {
        return make_dest_for_index_impl<Dest, ArraysTuple, K>(arrays, divisors, sizes, idx,
                                                              std::make_index_sequence<K>{});
    }

    template <typename Dest, typename... Arrays>
    consteval auto generate_combinations_impl(const Arrays&... arrays) {
        constexpr size_t K = sizeof...(Arrays);
        using ArraysTuple  = std::tuple<std::remove_cvref_t<Arrays>...>;
        const ArraysTuple               arrays_tuple{arrays...};
        constexpr std::array<size_t, K> sizes    = {array_size_v<Arrays>...};
        constexpr size_t                total    = prod_constexpr<array_size_v<Arrays>...>();
        const std::array<size_t, K>     divisors = make_divisors_from_sizes<K>(sizes);

        auto make_array = [&]<size_t... Is>(std::index_sequence<Is...>) consteval {
            return std::array<Dest, total>{make_dest_for_index<Dest>(arrays_tuple, divisors, sizes, Is)...};
        };

        return make_array(std::make_index_sequence<[](const size_t t) -> size_t { return t; }(total)>{});
    }

    template <typename Dest, typename... Arrays>
    consteval auto generate_combinations(const Arrays&... arrays) {
        static_assert(sizeof...(Arrays) >= 1, "Need at least one option array");
        return generate_combinations_impl<Dest, Arrays...>(arrays...);
    }

} // namespace chepp::nnue::tune

#endif // CHEPP_TUNE_H
