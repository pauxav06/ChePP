#ifndef CHEPP_NNUE_ACCUMULATOR_H
#define CHEPP_NNUE_ACCUMULATOR_H

#include "layer_base.h"
#include "utils.h"

#include <hwy/aligned_allocator.h>

namespace chepp::nnue::layers {
    template <typename IdxT, size_t IS, typename T, size_t OS>
        requires(std::is_unsigned_v<IdxT> && std::numeric_limits<IdxT>::max() >= IS)
    struct Accumulator {
        using index_t = IdxT;
        using value_t = T;

        static constexpr size_t input_size_v  = IS;
        static constexpr size_t output_size_v = OS;

        struct IExecutable {
            virtual ~IExecutable() = default;

            virtual void
            forward(const index_t*, size_t, value_t*) const noexcept = 0;

            virtual void
            forward_incremental(
                const value_t*, const index_t*, size_t, const index_t*, size_t, value_t*) const noexcept = 0;

            [[nodiscard]] virtual size_t
            padding() const noexcept {
                return size_t{0};
            }
        };

        using iexecutable_t = IExecutable;
        using ikernel_t     = IKernelBase<Accumulator>;

        struct Layer final : ILayerBase<Accumulator> {
            template <std::ranges::range WR, std::ranges::range BR>
            explicit Layer(WR&& weights, BR&& biases) noexcept
                : m_weights(std::ranges::begin(weights), std::ranges::end(weights)),
                  m_biases(std::ranges::begin(biases), std::ranges::end(biases)) {
                HWY_ASSERT(m_weights.size() == input_size_v * output_size_v);
                HWY_ASSERT(m_biases.size() == output_size_v);
            }

            [[nodiscard]] std::string
            name() const noexcept override {
                return fmt::format(
                    "name=Accumulator,input-type={},output-type={}", type_name_v<value_t>, type_name_v<value_t>);
            }

            [[nodiscard]] closure_type
            make_benchmark_closure(const std::shared_ptr<const ikernel_t>& kernel) const noexcept override {
                hwy::AlignedVector<index_t> inputs(std::max(static_cast<size_t>(bit::get_msb(input_size_v)), size_t{1}));
                std::iota(inputs.begin(), inputs.end(), static_cast<index_t>(0));
                hwy::AlignedVector<value_t> outputs(output_size_v + kernel->padding());
                size_t repetitions = std::max(size_t{128000} / (inputs.size() * output_size_v), size_t{1});

                return [=](hwy::FuncInput) mutable -> hwy::FuncOutput {
                    volatile hwy::FuncOutput out = hwy::Unpredictable1();
                    for (size_t i{0}; i < repetitions; ++i) {
                        kernel->forward(std::data(inputs), std::size(inputs), std::data(outputs));
                        out += static_cast<hwy::FuncOutput>(outputs.back());
                    }
                    return out;
                };
            }

            [[nodiscard]] HWY_INLINE std::span<const value_t>
                                     weights() const noexcept {
                return m_weights;
            }

            [[nodiscard]] HWY_INLINE std::span<const value_t>
                                     biases() const noexcept {
                return m_biases;
            }

          private:
            hwy::AlignedVector<value_t> m_weights;
            hwy::AlignedVector<value_t> m_biases;
        };

        template <std::input_or_output_iterator It>
        static std::shared_ptr<Layer>
        make_layer(It& begin, const It& end) noexcept {
            auto w_begin = reinterpret_cast<const value_t*>(&*begin);
            HWY_ASSERT(read_n<value_t>(begin, end, input_size_v * output_size_v));
            std::span<const value_t> w{w_begin, input_size_v * output_size_v};
            auto                     b_begin = reinterpret_cast<const value_t*>(&*begin);
            HWY_ASSERT(read_n<value_t>(begin, end, output_size_v));
            std::span<const value_t> b{b_begin, output_size_v};
            return std::make_shared<Layer>(w, b);
        }

        using layer_t = Layer;
    };

    struct AccumulatorSimd {
        size_t unroll;

        bool
        operator==(const AccumulatorSimd&) const = default;
    };

} // namespace chepp::nnue::layers

template <>
struct std::hash<chepp::nnue::layers::AccumulatorSimd> {
    size_t
    operator()(const chepp::nnue::layers::AccumulatorSimd& cfg) const noexcept {
        return cfg.unroll;
    }
};

#endif // UNTITLED3_AFFINE_SPARSE_H