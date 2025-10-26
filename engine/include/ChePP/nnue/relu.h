#ifndef CHEPP_RELU_H_
#define CHEPP_RELU_H_

#include "layers.h"
#include "types.h"
#include "utils.h"

namespace chepp::nnue::layers::relu {
    using Scalar = Kernel<0>;
    using Simd   = Kernel<1>;

    CHEPP_NNUE_DEFINE_INTEGER_CONSTANT_TYPE(Shift, int8_t)

    template <TypesConcept Types, ShiftConcept Shift>
    struct Layer : LayerBase<Layer<Types, Shift>> {
        using extent_type = size_t;
        using input_type  = Types::in;
        using output_type = Types::out;

        static_assert(std::is_integral_v<input_type> && std::is_integral_v<output_type>);
        static_assert(sizeof(output_type) < sizeof(input_type));

        struct Params {
            Dims dims;
        };

        explicit Layer(const Params params) : m_input_size(params.dims.in), m_output_size(params.dims.out) {
            HWY_ASSERT(m_input_size == m_output_size);
        }

        [[nodiscard]] HWY_INLINE extent_type
        input_size() const {
            return m_input_size;
        }
        [[nodiscard]] HWY_INLINE extent_type
        output_size() const {
            return m_output_size;
        }
        [[nodiscard]] HWY_INLINE static constexpr Shift_t
        shift() {
            return Shift::value;
        }
        [[nodiscard]] HWY_INLINE static constexpr input_type
        min() {
            return static_cast<input_type>(0);
        }
        [[nodiscard]] HWY_INLINE static constexpr input_type
        max() {
            return static_cast<input_type>(std::numeric_limits<output_type>::max());
        }

      private:
        const extent_type m_input_size{};
        const extent_type m_output_size{};
    };

} // namespace chepp::nnue::layers::relu

#endif // CHEPP_RELU_H
