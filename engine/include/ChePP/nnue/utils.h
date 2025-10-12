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
} // namespace chepp::nnue::utils

#endif
