#ifndef CHEPP_TESTS_UTILS_H_
#define CHEPP_TESTS_UTILS_H_

#include <random>
#include <stdint.h>

namespace chepp::tests::utils
{
    template <typename T>
    void fill_random(
        T* ptr,
        const size_t n,
        const T min = std::numeric_limits<T>::min(),
        const T max = std::numeric_limits<T>::max(),
        const unsigned seed = 1234)
    {
        std::mt19937                       rng(seed);
        std::uniform_int_distribution dist(min, max);
        for (size_t i = 0; i < n; ++i)
        {
            ptr[i] = static_cast<T>(dist(rng));
        }
    }

    template <typename T, typename Alloc>
    void fill_random(
        std::vector<T, Alloc>& v,
        const T min = std::numeric_limits<T>::min(),
        const T max = std::numeric_limits<T>::max(),
        unsigned seed = 1234)
    {
        fill_random(v.data(), v.size(), min, max, seed);
    }

}


#endif // CHEPP_UTILS_H
