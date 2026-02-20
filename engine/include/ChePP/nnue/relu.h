#ifndef CHEPP_RELU_H_
#define CHEPP_RELU_H_

#include "layers.h"
#include "utils.h"

#include <hwy/base.h>

namespace chepp::nnue::layers {
    template <typename InT, std::size_t IS, typename OutT, unsigned Q>
        requires(std::is_integral_v<InT> && std::is_integral_v<OutT> && std::is_signed_v<InT> &&
                 !std::is_signed_v<OutT> && sizeof(InT) >= sizeof(OutT) && utils::is_power_of_two(Q))
    struct ClippedReLULayer final : ILayer {
        using extent_type = size_t;
        using input_type  = InT;
        using output_type = OutT;

        static constexpr int reductions = sizeof(OutT) / sizeof(InT);
        static constexpr int quantize   = Q;
        static constexpr int shift      = std::bit_width(Q) - 1;

        static constexpr input_type min{static_cast<input_type>(0)};
        static constexpr input_type max{std::numeric_limits<std::make_signed_t<output_type>>::max()};

        [[nodiscard]] std::size_t static constexpr size() {
            return IS;
        }

        struct IKernel : KernelBase {
            virtual void
            forward(const input_type* input, output_type* output) const = 0;
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
            hwy::AlignedVector<input_type> inputs(size() + kernel.input_padding());
            fill_random(inputs);
            hwy::AlignedVector<output_type> outputs(size() + kernel.output_padding());
            /**
            return utils::benchmark([&] {
                kernel.forward(inputs.data(), outputs.data());
                return outputs.back();
            }, 0.01);
            **/
            hwy::FuncInput func_input[1]{1};
            hwy::Result    result[1]{};
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
    };

    struct ClippedReluSimd {
        int unroll;

        bool
        operator==(const ClippedReluSimd&) const = default;
    };

} // namespace chepp::nnue::layers

template <>
struct std::hash<chepp::nnue::layers::ClippedReluSimd> {
    std::size_t
    operator()(const chepp::nnue::layers::ClippedReluSimd& cfg) const noexcept {
        return cfg.unroll;
    }
};

#endif // CHEPP_RELU_H
