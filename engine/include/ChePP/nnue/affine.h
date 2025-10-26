#ifndef CHEPP_AFFINE_H
#define CHEPP_AFFINE_H

#include "hwy/targets.h"
#include "layers.h"
#include "types.h"
#include "utils.h"

#include <hwy/aligned_allocator.h>

namespace chepp::nnue::layers::affine {
    using namespace std::experimental;

    using Scalar     = Kernel<0>;
    using SimdColMaj = Kernel<2>;

    using SumOfMulQuadAcc = Operation<0>;
    using SumOfMulPairAcc = Operation<1>;

    template <TypesConcept Types>
    struct Layer : LayerBase<Layer<Types>> {
        using extent_type = std::size_t;
        using input_type  = Types::in;
        using output_type = Types::out;

        [[nodiscard]] HWY_INLINE extent_type
        input_size() const {
            return m_input_size;
        }
        [[nodiscard]] HWY_INLINE extent_type
        output_size() const {
            return m_output_size;
        }
        [[nodiscard]] HWY_INLINE std::span<const input_type>
                                 weights() const {
            return m_weights;
        }
        [[nodiscard]] HWY_INLINE std::span<const output_type>
                                 biases() const {
            return m_biases;
        }

        struct Params {
            Dims                         dims;
            std::span<const input_type>  weights;
            std::span<const output_type> biases;
        };

        explicit Layer(const Params params)
            : m_input_size(params.dims.in), m_output_size(params.dims.out),
              m_weights(params.weights.begin(), params.weights.end()),
              m_biases(params.biases.begin(), params.biases.end()) {
            HWY_ASSERT(m_weights.size() == m_input_size * m_output_size);
            HWY_ASSERT(m_biases.size() == m_output_size);
        }

      private:
        const extent_type m_input_size;
        const extent_type m_output_size;

        const hwy::AlignedVector<input_type>  m_weights;
        const hwy::AlignedVector<output_type> m_biases;
    };

} // namespace chepp::nnue::layers::affine

#endif // CHEPP_AFFINE_H
