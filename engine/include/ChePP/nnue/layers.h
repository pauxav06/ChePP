#ifndef CHEPP_NNUE_LAYERS_H_
#define CHEPP_NNUE_LAYERS_H_

#include "hwy/base.h"
#include "hwy/targets.h"
#include "meta.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

#define CHEPP_NNUE_DEFINE_INTEGER_CONSTANT_TYPE(NAME, BASE_TYPE)                                                       \
    struct NAME##_tag {};                                                                                              \
                                                                                                                       \
    using NAME##_t = BASE_TYPE;                                                                                        \
                                                                                                                       \
    template <NAME##_t V>                                                                                              \
    struct NAME : std::integral_constant<NAME##_t, V>, NAME##_tag {};                                                  \
                                                                                                                       \
    template <typename T>                                                                                              \
    concept NAME##Concept = std::derived_from<T, NAME##_tag> && requires {                                             \
        typename T::value_type;                                                                                        \
        { T::value } -> std::convertible_to<typename T::value_type>;                                                   \
    } && std::is_base_of_v<std::integral_constant<typename T::value_type, T::value>, T>;

namespace chepp::nnue::layers {
    CHEPP_NNUE_DEFINE_INTEGER_CONSTANT_TYPE(Kernel, uint8_t);
    CHEPP_NNUE_DEFINE_INTEGER_CONSTANT_TYPE(Operation, uint8_t);
    CHEPP_NNUE_DEFINE_INTEGER_CONSTANT_TYPE(Unroll, uint8_t);
    CHEPP_NNUE_DEFINE_INTEGER_CONSTANT_TYPE(Quantization, uint8_t);

    template <typename T>
    concept TypesConcept = requires {
        typename T::in;
        typename T::out;
    };

    template <typename In, typename Out>
    struct Types {
        using in  = In;
        using out = Out;
    };

    struct Dims {
        const size_t in;
        const size_t out;
    };

    template <typename C, typename T>
    concept BufferConstraintConcept = requires(C c, std::span<T> s) {
        { c.span_satisfies(s) } -> std::convertible_to<bool>;
        { C::merge(c, c) } -> std::same_as<C>;
    };

    struct SizeConstraint {
        std::size_t min_size = 0;

        template <typename T>
        constexpr bool
        span_satisfies(std::span<T> s) const noexcept {
            return s.size() >= min_size;
        }

        static constexpr SizeConstraint
        merge(SizeConstraint a, SizeConstraint b) noexcept {
            return {std::max(a.min_size, b.min_size)};
        }
    };

    struct AlignConstraint {
        std::size_t alignment = 1;

        template <typename T>
        constexpr bool
        span_satisfies(std::span<T> s) const noexcept {
            return std::bit_cast<std::uintptr_t>(s.data()) % alignment == 0;
        }

        static constexpr AlignConstraint
        merge(AlignConstraint a, AlignConstraint b) noexcept {
            return {std::max(a.alignment, b.alignment)};
        }
    };

    template <typename T, typename... Cs>
        requires(BufferConstraintConcept<Cs, T> && ...)
    struct BufferConstraintAggregate {
        using value_type = T;
        std::tuple<Cs...> constraints;

        constexpr BufferConstraintAggregate() noexcept : constraints(Cs{}...) {
        }

        constexpr BufferConstraintAggregate(Cs... cs) noexcept : constraints(cs...) {
        }

        template <typename... Args>
            requires(sizeof...(Args) == sizeof...(Cs)) && (std::constructible_from<Cs, Args> && ...)
        constexpr BufferConstraintAggregate(Args... args) noexcept : constraints(Cs{args}...) {
        }

        static constexpr auto
        merge(const BufferConstraintAggregate& a, const BufferConstraintAggregate& b) noexcept {
            return BufferConstraints(Cs::merge(std::get<Cs>(a.constraints), std::get<Cs>(b.constraints))...);
        }

        constexpr bool
        span_satisfies(std::span<T> s) const noexcept {
            return (std::get<Cs>(constraints).span_satisfies(s) && ...);
        }

        template <typename C>
        constexpr const auto&
        get() const noexcept {
            return std::get<C>(constraints);
        }

        template <typename C>
        constexpr const C&
        get(C) const noexcept {
            return std::get<C>(constraints);
        }

        static constexpr std::string
        to_string() noexcept {
            std::ostringstream oss;
            ((oss << "Constraint " << typeid(Cs).name() << "\n"), ...);
            return oss.str();
        }
    };

    template <typename T>
    using BufferConstraints = BufferConstraintAggregate<T, SizeConstraint, AlignConstraint>;

    template <typename T>
    concept LayerConcept = requires {
        typename T::input_type;
        typename T::output_type;
    };

    // A immutable conceptual layer describing an operation on matrices.
    // Used to create stateful layers that can perform that operation.
    // These stateful layers hold a shared pointer to this layer in case they need to be reconfigured at runtime
    // using the init() method
    template <typename Derived>
    struct LayerBase : std::enable_shared_from_this<Derived> {
        struct IState {
            using input_type  = Derived::input_type;
            using output_type = Derived::output_type;
            using input_bc    = BufferConstraints<const input_type>;
            using output_bc   = BufferConstraints<output_type>;

            virtual ~IState() = default;
            virtual void
            init() {
            }
            virtual HWY_INLINE input_bc
            input_buffer_constraints() const = 0;
            virtual HWY_INLINE output_bc
                         output_buffer_constraints() const                                  = 0;
            virtual void forward(std::span<const input_type>, std::span<output_type>) const = 0;
        };
    };

    template <typename Layer>
    std::shared_ptr<Layer>
    make_layer(const typename Layer::Params params) {
        return std::make_shared<Layer>(params);
    }

#define CHEPP_BEFORE_LAYER()                                                                                           \
    template <typename...>                                                                                             \
    struct State;

// bridge between global and highway namespace
#define CHEPP_AFTER_LAYER()                                                                                            \
    template <typename Layer, typename... Config>                                                                      \
    std::unique_ptr<typename Layer::IState> make_state(std::shared_ptr<Layer> layer, Config...) {                      \
        return std::make_unique<State<Layer, Config...>>(layer);                                                       \
    }                                                                                                                  \
    template <typename Layer, typename... Config>                                                                      \
    std::unique_ptr<typename Layer::IState> make_state(std::shared_ptr<Layer> layer, std::tuple<Config...>) {          \
        return std::make_unique<State<Layer, Config...>>(layer);                                                       \
    }
} // namespace chepp::nnue::layers

#endif
