#ifndef CHEPP_NNUE_META_H
#define CHEPP_NNUE_META_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace chepp::nnue::meta {
    template <typename Tuple1, typename Tuple2>
    struct tuple_cat_types;

    template <typename... Ts, typename... Us>
    struct tuple_cat_types<std::tuple<Ts...>, std::tuple<Us...>> {
        using type = std::tuple<Ts..., Us...>;
    };

    template <typename Tuple1, typename Tuple2>
    using tuple_cat_types_t = tuple_cat_types<Tuple1, Tuple2>::type;

    // This method of generating cartesian product uses a constexpr array mapping table
    // This circumvents the problems of templating recursion depth going out of hands and keeps
    // Compile times very low.
    template <typename... Tuples>
    struct Cartesian {
      private:
        static_assert(sizeof...(Tuples) >= 1, "Need at least one tuple");
        static constexpr std::size_t                N       = sizeof...(Tuples);
        static constexpr std::array<std::size_t, N> sizes   = {std::tuple_size_v<Tuples>...};
        static constexpr std::size_t                product = ((std::tuple_size_v<Tuples>)*...);

        static constexpr auto make_index_table() {
            std::array<std::array<std::size_t, N>, (product == 0 ? 1 : product)> table{};
            for (std::size_t k = 0; k < product; ++k) {
                std::size_t v = k;
                for (std::size_t i = N; i-- > 0;) {
                    table[k][i] = v % sizes[i];
                    v /= sizes[i];
                }
            }
            return table;
        }

        static constexpr auto index_table = make_index_table();

        template <std::array<std::size_t, N> Indices, typename... Ts>
        struct map_indices_impl {
            template <std::size_t... I>
            static auto helper(std::index_sequence<I...>) -> std::tuple<std::tuple_element_t<Indices[I], Ts>...>;

            using type = decltype(helper(std::make_index_sequence<sizeof...(Ts)>{}));
        };

        template <auto... IndexArrays>
        struct build {
            using type = std::tuple<typename map_indices_impl<IndexArrays, Tuples...>::type...>;
        };

        template <std::size_t... Ks>
        static auto make_type_helper(std::index_sequence<Ks...>) -> build<index_table[Ks]...>::type;

      public:
        using type = decltype(make_type_helper(std::make_index_sequence<product>{}));
    };

    template <template <typename...> class Target, typename Tuple>
    struct unpack_tuple;

    template <template <typename...> class Target, typename... Ts>
    struct unpack_tuple<Target, std::tuple<Ts...>> {
        using type = Target<Ts...>;
    };

    template <template <typename> class F, typename Tuple>
    struct map_tuple;

    template <template <typename> class F, typename... Ts>
    struct map_tuple<F, std::tuple<Ts...>> {
        using type = std::tuple<typename F<Ts>::type...>;
    };

    template <typename... Tuples>
    using Cartesian_t = Cartesian<Tuples...>::type;

    template <typename Tuple, typename F, std::size_t... I>
    constexpr void for_each_type_impl(F&& f, std::index_sequence<I...>) {
        (f.template operator()<std::tuple_element_t<I, Tuple>>(), ...);
    }

    template <typename Tuple, typename F>
    constexpr void for_each_type(F&& f) {
        for_each_type_impl<Tuple>(std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }

    template <typename Tuple, typename F, std::size_t... I>
    constexpr void for_each_in_tuple_impl(Tuple&& t, F&& f, std::index_sequence<I...>) {
        (f(std::get<I>(t), std::integral_constant<std::size_t, I>{}), ...);
    }

    template <typename Tuple, typename F>
    constexpr void for_each_in_tuple(Tuple&& t, F&& f) {
        constexpr auto N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
        for_each_in_tuple_impl(std::forward<Tuple>(t), std::forward<F>(f), std::make_index_sequence<N>{});
    }

    template <template <typename> class F, typename List>
    struct apply_template;

    template <template <typename> class F, typename... Ts>
    struct apply_template<F, std::tuple<Ts...>> {
        using type = std::tuple<F<Ts>...>;
    };

    template <template <typename> class F, typename... Ts>
    using apply_template_t = apply_template<F, Ts...>::type;

    template <typename Tuple, std::size_t... Is>
    constexpr auto instantiate_tuple_impl(std::index_sequence<Is...>) {
        return std::tuple<std::decay_t<std::tuple_element_t<Is, Tuple>>...>{std::tuple_element_t<Is, Tuple>{}...};
    }

    template <typename Tuple>
    constexpr auto instantiate_tuple() {
        constexpr std::size_t N = std::tuple_size_v<Tuple>;
        return instantiate_tuple_impl<Tuple>(std::make_index_sequence<N>{});
    }

    template <typename... Tuples>
    constexpr auto instantiate_all(std::tuple<Tuples...>) {
        return std::array{instantiate_tuple<Tuples>()...};
    }

    enum class ScalarType : uint8_t {
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float32,
        Float64,
    };

    template <ScalarType>
    struct scalar_type_traits;

    template <>
    struct scalar_type_traits<ScalarType::Int8> {
        using type                                  = int8_t;
        static constexpr const char* name           = "int8";
        static constexpr const char* cpp_type       = "int8_t";
        static constexpr const char* literal_suffix = "";
    };
    template <>
    struct scalar_type_traits<ScalarType::Int16> {
        using type                                  = int16_t;
        static constexpr const char* name           = "int16";
        static constexpr const char* cpp_type       = "int16_t";
        static constexpr const char* literal_suffix = "";
    };
    template <>
    struct scalar_type_traits<ScalarType::Int32> {
        using type                                  = int32_t;
        static constexpr const char* name           = "int32";
        static constexpr const char* cpp_type       = "int32_t";
        static constexpr const char* literal_suffix = "";
    };
    template <>
    struct scalar_type_traits<ScalarType::Int64> {
        using type                                  = int64_t;
        static constexpr const char* name           = "int64";
        static constexpr const char* cpp_type       = "int64_t";
        static constexpr const char* literal_suffix = "LL";
    };

    template <>
    struct scalar_type_traits<ScalarType::UInt8> {
        using type                                  = uint8_t;
        static constexpr const char* name           = "uint8";
        static constexpr const char* cpp_type       = "uint8_t";
        static constexpr const char* literal_suffix = "U";
    };
    template <>
    struct scalar_type_traits<ScalarType::UInt16> {
        using type                                  = uint16_t;
        static constexpr const char* name           = "uint16";
        static constexpr const char* cpp_type       = "uint16_t";
        static constexpr const char* literal_suffix = "U";
    };
    template <>
    struct scalar_type_traits<ScalarType::UInt32> {
        using type                                  = uint32_t;
        static constexpr const char* name           = "uint32";
        static constexpr const char* cpp_type       = "uint32_t";
        static constexpr const char* literal_suffix = "U";
    };
    template <>
    struct scalar_type_traits<ScalarType::UInt64> {
        using type                                  = uint64_t;
        static constexpr const char* name           = "uint64";
        static constexpr const char* cpp_type       = "uint64_t";
        static constexpr const char* literal_suffix = "ULL";
    };

    template <>
    struct scalar_type_traits<ScalarType::Float32> {
        using type                                  = float;
        static constexpr const char* name           = "float32";
        static constexpr const char* cpp_type       = "float";
        static constexpr const char* literal_suffix = "f";
    };
    template <>
    struct scalar_type_traits<ScalarType::Float64> {
        using type                                  = double;
        static constexpr const char* name           = "float64";
        static constexpr const char* cpp_type       = "double";
        static constexpr const char* literal_suffix = "";
    };

    template <ScalarType T>
    using scalar_t = scalar_type_traits<T>::type;

    template <typename>
    struct scalar_enum_traits;

#define DEFINE_SCALAR_ENUM_TRAIT(cpp_type, enum_value)                                                                 \
    template <>                                                                                                        \
    struct scalar_enum_traits<cpp_type> {                                                                              \
        static constexpr ScalarType value = enum_value;                                                                \
        using traits                      = scalar_type_traits<enum_value>;                                            \
    };

    DEFINE_SCALAR_ENUM_TRAIT(int8_t, ScalarType::Int8)
    DEFINE_SCALAR_ENUM_TRAIT(int16_t, ScalarType::Int16)
    DEFINE_SCALAR_ENUM_TRAIT(int32_t, ScalarType::Int32)
    DEFINE_SCALAR_ENUM_TRAIT(int64_t, ScalarType::Int64)
    DEFINE_SCALAR_ENUM_TRAIT(uint8_t, ScalarType::UInt8)
    DEFINE_SCALAR_ENUM_TRAIT(uint16_t, ScalarType::UInt16)
    DEFINE_SCALAR_ENUM_TRAIT(uint32_t, ScalarType::UInt32)
    DEFINE_SCALAR_ENUM_TRAIT(uint64_t, ScalarType::UInt64)
    DEFINE_SCALAR_ENUM_TRAIT(float, ScalarType::Float32)
    DEFINE_SCALAR_ENUM_TRAIT(double, ScalarType::Float64)

#undef DEFINE_SCALAR_ENUM_TRAIT

    constexpr const char* to_string(ScalarType t) noexcept {
        switch (t) {
            case ScalarType::Int8:
                return "int8";
            case ScalarType::Int16:
                return "int16";
            case ScalarType::Int32:
                return "int32";
            case ScalarType::Int64:
                return "int64";
            case ScalarType::UInt8:
                return "uint8";
            case ScalarType::UInt16:
                return "uint16";
            case ScalarType::UInt32:
                return "uint32";
            case ScalarType::UInt64:
                return "uint64";
            case ScalarType::Float32:
                return "float32";
            case ScalarType::Float64:
                return "float64";
        }
        return "unknown";
    }
} // namespace chepp::nnue::meta

#endif // CHEPP_META_H
