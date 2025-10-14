#ifndef CHEPP_NNUE_LAYERS_H_
#define CHEPP_NNUE_LAYERS_H_

#pragma once
#include <cstdint>
#include <type_traits>

namespace meta {

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
    using scalar_t = typename scalar_type_traits<T>::type;

    template <typename>
    struct scalar_enum_traits;

#define DEFINE_SCALAR_ENUM_TRAIT(cpp_type, enum_value)                                                                 \
    template <>                                                                                                        \
    struct scalar_enum_traits<cpp_type> {                                                                              \
        static constexpr ScalarType value = enum_value;                                                                \
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

} // namespace meta

#endif
