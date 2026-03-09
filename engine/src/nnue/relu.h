#ifndef CHEPP_RELU_H_
#define CHEPP_RELU_H_

#include "layer_base.h"
#include "utils.h"

#include <hwy/base.h>

namespace chepp::nnue::layers {
    template <typename InT, size_t IS, typename OutT, unsigned Q>
        requires(std::is_integral_v<InT> && std::is_integral_v<OutT> && std::is_signed_v<InT> &&
                 !std::is_signed_v<OutT> && sizeof(InT) >= sizeof(OutT) && is_power_of_two(Q))
    struct ClippedRelu {
        using input_t  = InT;
        using output_t = OutT;

        static constexpr size_t size_v     = IS;
        static constexpr int    reductions = sizeof(OutT) / sizeof(InT);
        static constexpr int    quantize   = Q;
        static constexpr int    shift      = std::bit_width(Q) - 1;

        static constexpr input_t min{static_cast<input_t>(0)};
        static constexpr input_t max{std::numeric_limits<std::make_signed_t<output_t>>::max()};

        struct IExecutable {
            virtual ~IExecutable() = default;
            virtual void
            forward(const input_t*, output_t*) const noexcept = 0;
            [[nodiscard]] virtual size_t
            input_padding() const noexcept {
                return size_t{0};
            }
            [[nodiscard]] virtual size_t
            output_padding() const noexcept {
                return size_t{0};
            }
        };

        using iexecutable_t = IExecutable;
        using ikernel_t     = IKernelBase<ClippedRelu>;

        struct Layer final : ILayerBase<ClippedRelu> {
            explicit Layer(empty_t) noexcept {};
            Layer() = default;

            [[nodiscard]] std::string
            name() const noexcept override {
                return fmt::format(
                    "clipped-relu,input_type={},output_type={}", type_name_v<input_t>, type_name_v<output_t>);
            }

            [[nodiscard]] closure_type
            make_benchmark_closure(const std::shared_ptr<const ikernel_t>& kernel) const noexcept override {
                hwy::AlignedVector<input_t> inputs(size_v + kernel->input_padding());
                fill_random(inputs, 0);
                hwy::AlignedVector<output_t> outputs(size_v + kernel->output_padding());

                return [=](hwy::FuncInput) mutable -> hwy::FuncOutput {
                    kernel->forward(std::data(inputs), std::data(outputs));
                    return static_cast<hwy::FuncOutput>(outputs.back());
                };
            }
        };

        template <std::input_or_output_iterator It>
        static std::shared_ptr<Layer>
        make_layer(It&, const It&) {
            return std::make_shared<Layer>();
        }

        using layer_t = Layer;
    };

    struct ClippedReluSimd {
        size_t unroll;

        bool
        operator==(const ClippedReluSimd&) const = default;
    };

} // namespace chepp::nnue::layers

template <>
struct std::hash<chepp::nnue::layers::ClippedReluSimd> {
    size_t
    operator()(const chepp::nnue::layers::ClippedReluSimd& cfg) const noexcept {
        return cfg.unroll;
    }
};

#endif // CHEPP_RELU_H
