#ifndef CHEPP_TESTS_UTILS_H_
#define CHEPP_TESTS_UTILS_H_

#include <iterator>
#include <limits>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace chepp::tests::utils {

    template <typename T>
    void fill_random(T*       ptr,
                     size_t   n,
                     T        min  = std::numeric_limits<T>::min(),
                     T        max  = std::numeric_limits<T>::max(),
                     unsigned seed = 1234) {
        std::mt19937 rng(seed);

        using dist_t = std::
            conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

        dist_t dist(min, max);

        for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<T>(dist(rng));
    }

    template <typename Container>
    void fill_random(Container&                     c,
                     typename Container::value_type min  = std::numeric_limits<typename Container::value_type>::min(),
                     typename Container::value_type max  = std::numeric_limits<typename Container::value_type>::max(),
                     unsigned                       seed = 1234) {
        using T = Container::value_type;
        std::mt19937 rng(seed);

        using dist_t = std::
            conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

        dist_t dist(min, max);

        for (auto& x : c) x = static_cast<T>(dist(rng));
    }
} // namespace chepp::tests::utils

#endif // CHEPP_UTILS_H
