#ifndef CHEPP_NNUE_UTILS_H_
#define CHEPP_NNUE_UTILS_H_

#include <hwy/aligned_allocator.h>

#include <array>
#include <cstddef>
#include <experimental/mdspan>
#include <experimental/mdarray>

namespace chepp::nnue::utils
{
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
} // namespace chepp::nnue::utils

#endif
