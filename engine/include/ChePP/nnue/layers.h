#ifndef CHEPP_NNUE_LAYERS_H_
#define CHEPP_NNUE_LAYERS_H_

#include "meta.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <type_traits>

namespace chepp::nnue::layers {
    struct kernel_tag {};
    struct operation_tag {};
    struct unroll_tag {};
    struct quantization_tag {};

    using kernel_t       = uint8_t;
    using op_t           = uint8_t;
    using unroll_t       = uint8_t;
    using quantization_t = uint8_t;

    template <kernel_t V>
    struct Kernel : std::integral_constant<kernel_t, V>, kernel_tag {};

    template <op_t V>
    struct Operation : std::integral_constant<op_t, V>, operation_tag {};

    template <unroll_t V>
    struct Unroll : std::integral_constant<unroll_t, V>, unroll_tag {};

    template <quantization_t V>
        requires(std::popcount(V) == 1)
    struct Quantization : std::integral_constant<quantization_t, V>, quantization_tag {
        static constexpr auto shift = std::bit_width(V);
    };

    template <typename T>
    concept IntegralConstantConcept = requires {
        typename T::value_type;
        { T::value } -> std::convertible_to<typename T::value_type>;
    } && std::is_base_of_v<std::integral_constant<typename T::value_type, T::value>, T>;

    template <typename T>
    concept KernelConcept = std::derived_from<T, kernel_tag> && IntegralConstantConcept<T>;

    template <typename T>
    concept OperationConcept = std::derived_from<T, operation_tag> && IntegralConstantConcept<T>;

    template <typename T>
    concept UnrollConcept = std::derived_from<T, unroll_tag> && IntegralConstantConcept<T>;

    template <typename T>
    concept TypesConcept = requires {
        typename T::in;
        typename T::out;
    };

    template <typename In, typename Out>
    struct Types {
        using in  = In;
        using out = Out;

        static std::string to_string() {
            return "Types: [" + std::string{meta::scalar_enum_traits<in>::traits::name} + ", " +
                   std::string{meta::scalar_enum_traits<out>::traits::name} + "]";
        }
    };

    template <KernelConcept K, TypesConcept T, typename... Opts>
    using ParamComb_t = meta::Cartesian<std::tuple<K>, std::tuple<T>, Opts...>::type;

    struct Dims {
        const size_t in;
        const size_t out;
    };
} // namespace chepp::nnue::layers

#endif
