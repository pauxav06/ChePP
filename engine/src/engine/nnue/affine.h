#ifndef CHEPP_AFFINE_H
#define CHEPP_AFFINE_H

#include "hwy/targets.h"
#include "layer_base.h"
#include "utils.h"

#include <hwy/aligned_allocator.h>

namespace chepp::nnue::layers {
    template <typename InT, std::size_t IS, typename OutT, std::size_t OS, typename WT, typename BT>
    struct AffineLayer final : ILayer {
        using input_type   = InT;
        using output_type  = OutT;
        using weights_type = WT;
        using biases_type  = BT;

        explicit AffineLayer(std::span<const weights_type> weights, std::span<const biases_type> biases)
            : m_weights(std::from_range, weights), m_biases(std::from_range, biases) {
            HWY_ASSERT(m_weights.size() == input_size() * output_size());
            HWY_ASSERT(m_biases.size() == output_size());
        }

        [[nodiscard]] std::string name() const override {
            return format_error("Affine: ", type_name_v<input_type>, " -> ", type_name_v<output_type>);
        }

        [[nodiscard]] HWY_INLINE extent_type static constexpr input_size() {
            return IS;
        }

        [[nodiscard]] HWY_INLINE extent_type static constexpr output_size() {
            return OS;
        }

        [[nodiscard]] HWY_INLINE std::span<const weights_type>
                                 weights() const {
            return m_weights;
        }
        [[nodiscard]] HWY_INLINE std::span<const biases_type>
                                 biases() const {
            return m_biases;
        }

        struct IKernel : KernelBase {
            virtual void
            forward(const input_type*, output_type*) const = 0;
            [[nodiscard]] virtual std::size_t
            input_padding() const {
                return 0;
            }
            [[nodiscard]] virtual std::size_t
            output_padding() const {
                return 0;
            }
        };

        [[nodiscard]] double
        benchmark(const KernelBase& ref) const override {
            const auto&                    kernel = dynamic_cast<const IKernel&>(ref);
            hwy::AlignedVector<input_type> inputs(input_size() + kernel.input_padding());
            fill_random(inputs, 0);
            hwy::AlignedVector<output_type> outputs(output_size() + kernel.output_padding());

            hwy::FuncInput func_input[]{1};
            hwy::Result    result[std::size(func_input)] {};
            hwy::Params    params{};
            params.verbose           = false;
            params.target_rel_mad    = 1;
            params.precision_divisor = 1;
            params.seconds_per_eval  = 4e-5;
            if (!hwy::MeasureClosure(
                    [&](unsigned int) -> hwy::FuncOutput {
                        kernel.forward(inputs.data(), outputs.data());
                        return outputs.back();
                    },
                    func_input,
                    std::size(func_input),
                    result,
                    params)) {
                return std::numeric_limits<double>::max();
            }
            return result->ticks;
        }

      private:
        std::vector<weights_type> m_weights;
        std::vector<output_type>  m_biases;
    };

    enum class AffineOperation { SumOfMulQuadAdd, MulPairwiseAdd };

    struct AffineSimdColMaj {
        int             unroll;
        AffineOperation operation;

        bool
        operator==(const AffineSimdColMaj&) const = default;
    };

    struct AffineSimdRowMaj {
        int             unroll;
        AffineOperation operation;

        bool
        operator==(const AffineSimdRowMaj&) const = default;
    };

} // namespace chepp::nnue::layers

template <>
struct std::hash<chepp::nnue::layers::AffineSimdColMaj> {
    std::size_t
    operator()(const chepp::nnue::layers::AffineSimdColMaj& config) const noexcept {
        return static_cast<size_t>(config.unroll) ^ static_cast<size_t>(config.operation);
    }
};

template <>
struct std::hash<chepp::nnue::layers::AffineSimdRowMaj> {
    std::size_t
    operator()(const chepp::nnue::layers::AffineSimdRowMaj& config) const noexcept {
        return static_cast<size_t>(config.unroll) ^ static_cast<size_t>(config.operation);
    }
};

#endif // CHEPP_AFFINE_H
