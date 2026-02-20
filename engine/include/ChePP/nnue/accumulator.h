#ifndef UNTITLED3_AFFINE_SPARSE_H
#define UNTITLED3_AFFINE_SPARSE_H

#include "hwy/targets.h"
#include "layers.h"
#include "utils.h"

#include <hwy/aligned_allocator.h>

namespace chepp::nnue::layers {
    template <typename IdxT, std::size_t IS, typename T, std::size_t OS>
        requires(std::is_unsigned_v<IdxT> && std::numeric_limits<IdxT>::max() >= IS)
    struct AccumulatorLayer final : ILayer {
        using index_type = IdxT;
        using value_type = T;

        explicit AccumulatorLayer(std::span<const value_type> weights, std::span<const value_type> biases)
            : m_weights(std::from_range, weights), m_biases(std::from_range, biases) {
            HWY_ASSERT(m_weights.size() == input_size() * output_size());
            HWY_ASSERT(m_biases.size() == output_size());
        }

        [[nodiscard]] HWY_INLINE extent_type static constexpr input_size() {
            return IS;
        }

        [[nodiscard]] HWY_INLINE extent_type static constexpr output_size() {
            return OS;
        }

        [[nodiscard]] HWY_INLINE std::span<const value_type>
                                 weights() const {
            return m_weights;
        }
        [[nodiscard]] HWY_INLINE std::span<const value_type>
                                 biases() const {
            return m_biases;
        }

        struct IKernel : KernelBase {
            virtual void
            forward(const index_type*, std::size_t, value_type*) const = 0;
            virtual void
            forward_incremental(const value_type*,
                                const index_type*,
                                std::size_t,
                                const index_type*,
                                std::size_t,
                                value_type*) const = 0;
            [[nodiscard]] virtual std::size_t
            padding() const {
                return 0;
            }
        };

        [[nodiscard]] double
        benchmark(const KernelBase& ref) const override {
            const auto&                    kernel = dynamic_cast<const IKernel&>(ref);
            hwy::AlignedVector<index_type> inputs(input_size() / 100);
            std::ranges::iota(inputs, 0);
            hwy::AlignedVector<value_type> outputs(output_size() + kernel.padding());
            hwy::FuncInput                 func_input[1]{1};
            hwy::Result                    result[1]{};
            hwy::Params                    params{};
            params.verbose           = false;
            params.target_rel_mad    = 1;
            params.precision_divisor = 1;
            params.seconds_per_eval  = 4e-5;
            if (!hwy::MeasureClosure(
                    [&](unsigned int) -> hwy::FuncOutput {
                        kernel.forward(std::data(inputs), std::size(inputs), std::data(outputs));
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
        hwy::AlignedVector<value_type> m_weights;
        hwy::AlignedVector<value_type> m_biases;
    };

    struct AccumulatorSimd {
        int unroll;

        bool
        operator==(const AccumulatorSimd&) const = default;
    };

} // namespace chepp::nnue::layers

template <>
struct std::hash<chepp::nnue::layers::AccumulatorSimd> {
    std::size_t
    operator()(const chepp::nnue::layers::AccumulatorSimd& cfg) const noexcept {
        return cfg.unroll;
    }
};

#endif // UNTITLED3_AFFINE_SPARSE_H