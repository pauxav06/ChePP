#ifndef CHEPP_NNUE_UTILS_H_
#define CHEPP_NNUE_UTILS_H_

#include <hwy/aligned_allocator.h>

#include <array>
#include <experimental/mdarray>
#include <experimental/mdspan>
#include <random>

namespace chepp::nnue::utils {
    using namespace std::experimental;

    template <typename T>
        requires std::is_integral_v<T>
    constexpr T pad_up(const T n, const T m) noexcept {
        return ((n + m - 1) / m) * m;
    }

    template <typename T>
        requires std::is_integral_v<T>
    constexpr T pad_down(const T x, const T n) noexcept {
        if (n == 0) return x;
        return (x >= 0) ? (x / n) * n : ((x - n + 1) / n) * n;
    }

    template <typename T>
        requires std::is_integral_v<T>
    constexpr bool is_power_of_two(const T x) {
        return x != 0 && (x & (x - 1)) == 0;
    }

    using extent_type = std::size_t;

    template <extent_type e>
    struct extent_wrapper {
        static constexpr extent_type extent = e;
        extent_type                  value;
        constexpr                    operator extent_type() const noexcept { return value; }
    };

    template <typename... Wrapper>
    using make_extents_t = std::extents<extent_type, Wrapper::extent...>;

    template <typename... Wrapper>
    constexpr make_extents_t<Wrapper...> make_extents_from_values(const Wrapper&... w) {
        return make_extents_t<Wrapper...>{(w.value)...};
    }
    template <typename T, typename... Wrapper>
    struct Tensor {
        using value_type        = T;
        using extent_t          = decltype(make_extents_from_values(std::declval<Wrapper>()...));
        using span_t            = mdspan<T, extent_t>;
        using const_span_t      = mdspan<const T, extent_t>;
        using flat_span_t       = std::span<T>;
        using const_flat_span_t = std::span<const T>;

        extent_t s_extent;

        constexpr explicit Tensor(const Wrapper&... w) : s_extent(make_extents_from_values(w...)) {}

        constexpr const extent_t& extent() const noexcept { return s_extent; }

        template <typename Ptr>
        constexpr span_t make_span(Ptr* data) const {
            return span_t(data, s_extent);
        }

        template <typename Ptr>
        constexpr const_span_t make_const_span(const Ptr* data) const {
            return const_span_t(data, s_extent);
        }

        template <typename Ptr>
        constexpr flat_span_t make_flat_span(Ptr* data) const {
            return flat_span_t(data, s_extent.size());
        }

        template <typename Ptr>
        constexpr const_flat_span_t make_const_flat_span(const Ptr* data) const {
            return const_flat_span_t(data, s_extent.size());
        }
    };

    template <typename T, typename... Wrapper>
    auto make_tensor(Wrapper... wrappers) -> Tensor<T, Wrapper...>;

    template <typename T, typename... Wrapper>
    using Tensor_t = Tensor<T, Wrapper...>;

    template <typename T>
    hwy::AlignedVector<T> make_aligned_vector(std::span<T> s) {
        return hwy::AlignedVector<T>(s.begin(), s.end());
    }
} // namespace chepp::nnue::utils

#endif
