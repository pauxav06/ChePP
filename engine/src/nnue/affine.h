#ifndef CHEPP_AFFINE_H
#define CHEPP_AFFINE_H

#include "layer_base.h"
#include "utils.h"

#include <hwy/aligned_allocator.h>

namespace chepp::nnue::layers {
    template <typename InT, std::size_t IS, typename OutT, std::size_t OS, typename WT, typename BT>
    struct Affine {
        using input_t  = InT;
        using output_t = OutT;
        using weight_t = WT;
        using bias_t   = BT;

        static constexpr std::size_t input_size_v  = IS;
        static constexpr std::size_t output_size_v = OS;

        struct IExecutable {
            virtual ~IExecutable() = default;
            virtual void
            forward(const input_t*, output_t*) const = 0;

            [[nodiscard]] virtual std::size_t
            input_padding() const noexcept {
                return size_t{0};
            }

            [[nodiscard]] virtual std::size_t
            output_padding() const noexcept {
                return size_t{0};
            }
        };

        using iexecutable_t = IExecutable;
        using ikernel_t     = IKernelBase<Affine>;

        struct Layer final : ILayerBase<Affine> {
            explicit Layer(empty_t) noexcept : m_weights(input_size_v * output_size_v), m_biases(output_size_v){};

            template <std::ranges::range WR, std::ranges::range BR>
            explicit Layer(WR&& weights, BR&& biases) noexcept
                : m_weights(std::ranges::begin(weights), std::ranges::end(weights)),
                  m_biases(std::ranges::begin(biases), std::ranges::end(biases)) {
                HWY_ASSERT(m_weights.size() == input_size_v * output_size_v);
                HWY_ASSERT(m_biases.size() == output_size_v);
            }

            [[nodiscard]] std::string
            name() const noexcept override {
                return fmt::format("affine,input-type={},output-type={}", type_name_v<input_t>, type_name_v<output_t>);
            }

            [[nodiscard]] HWY_INLINE std::span<const weight_t>
                                     weights() const noexcept {
                return m_weights;
            }
            [[nodiscard]] HWY_INLINE std::span<const bias_t>
                                     biases() const noexcept {
                return m_biases;
            }

            [[nodiscard]] closure_type
            make_benchmark_closure(const std::shared_ptr<const ikernel_t>& kernel) const noexcept override {
                hwy::AlignedVector<input_t> inputs(input_size_v + kernel->input_padding());
                fill_random(inputs, 0);
                hwy::AlignedVector<output_t> outputs(output_size_v + kernel->output_padding());
                size_t repetitions = std::max(size_t{128000} / (input_size_v * output_size_v), size_t{1});

                return [=](hwy::FuncInput) mutable -> hwy::FuncOutput {
                    volatile hwy::FuncOutput out = hwy::Unpredictable1();
                    for (size_t i{0}; i < repetitions; ++i) {
                        kernel->forward(std::data(inputs), std::data(outputs));
                        out += static_cast<hwy::FuncOutput>(outputs.back());
                    }
                    return out;
                };
            }

          private:
            std::vector<weight_t> m_weights;
            std::vector<output_t> m_biases;
        };

        template <std::input_or_output_iterator It>
        static std::shared_ptr<Layer>
        make_layer(It& begin, const It& end) noexcept {
            auto w_begin = reinterpret_cast<const weight_t*>(&*begin);
            HWY_ASSERT(read_n<weight_t>(begin, end, input_size_v * output_size_v));
            std::span<const weight_t> w{w_begin, input_size_v * output_size_v};
            auto                      b_begin = reinterpret_cast<const bias_t*>(&*begin);
            HWY_ASSERT(read_n<bias_t>(begin, end, output_size_v));
            std::span<const bias_t> b{b_begin, output_size_v};
            return std::make_shared<Layer>(w, b);
        }

        using layer_t = Layer;
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

namespace ns = chepp::nnue::layers;

template <>
struct std::hash<ns::AffineSimdColMaj> {
    std::size_t
    operator()(const ns::AffineSimdColMaj& config) const noexcept {
        return static_cast<size_t>(config.unroll) ^ static_cast<size_t>(config.operation);
    }
};

template <>
struct std::hash<ns::AffineSimdRowMaj> {
    std::size_t
    operator()(const ns::AffineSimdRowMaj& config) const noexcept {
        return static_cast<size_t>(config.unroll) ^ static_cast<size_t>(config.operation);
    }
};

#endif // CHEPP_AFFINE_H
