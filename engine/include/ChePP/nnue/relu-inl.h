#include "relu.h"

#include <any>
#include <cmath>
#include <experimental/mdarray>
#include <experimental/mdspan>
#include <hwy/base.h>

#if defined(CHEPP_RELU_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_RELU_INL_H_
#undef CHEPP_RELU_INL_H_
#else
#define CHEPP_RELU_INL_H_
#endif

#include "utils-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers::relu {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;

        using namespace std::experimental;

        CHEPP_BEFORE_LAYER()

        template <typename Layer, KernelConcept Kernel>
        struct State<Layer, Kernel> : Layer::IState {
            using extent_type = Layer::extent_type;
            using input_type  = Layer::input_type;
            using output_type = Layer::output_type;

            explicit State(std::shared_ptr<const Layer> layer) : m_layer(layer) {
            }

            [[nodiscard]] HWY_INLINE BufferConstraints<const input_type>
                                     input_buffer_constraints() const override {
                return {m_layer->input_size(), sizeof(input_type)};
            }
            [[nodiscard]] HWY_INLINE BufferConstraints<output_type>
                                     output_buffer_constraints() const override {
                return {m_layer->output_size(), sizeof(output_type)};
            }

            void
            forward(std::span<const input_type> input_view, std::span<output_type> output_view) const override {
                HWY_ASSERT(input_view.size() >= m_layer->input_size());
                HWY_ASSERT(output_view.size() >= m_layer->output_size());

                auto* HWY_RESTRICT input_ptr  = input_view.data();
                auto* HWY_RESTRICT output_ptr = output_view.data();

                std::span<const input_type> input{input_ptr, m_layer->input_size()};
                std::span<output_type>      output{output_ptr, m_layer->output_size()};

                for (extent_type i = 0; i < input.size(); ++i) {
                    input_type val = input[i];
                    if constexpr (Layer::shift() != 0) val >>= Layer::shift();
                    val       = std::max(Layer::min(), val);
                    val       = std::min(Layer::max(), val);
                    output[i] = static_cast<output_type>(val);
                }
            }

          private:
            const std::shared_ptr<const Layer> m_layer;
        };

        template <typename Layer, UnrollConcept U>
        struct State<Layer, Simd, U> : Layer::IState {
            using extent_type = Layer::extent_type;
            using input_type  = Layer::input_type;
            using output_type = Layer::output_type;

            [[nodiscard]] HWY_INLINE static constexpr extent_type
            factor() {
                return sizeof(input_type) / sizeof(output_type);
            }
            [[nodiscard]] HWY_INLINE static constexpr extent_type
            unroll() {
                return U::value;
            }

            using Din  = hn::ScalableTag<input_type>;
            using Vin  = hn::VFromD<Din>;
            using Dout = hn::ScalableTag<output_type>;
            using Vout = hn::VFromD<Dout>;

            [[nodiscard]] HWY_INLINE static HWY_LANES_CONSTEXPR extent_type
            in_lanes() {
                return hn::Lanes(Din());
            }
            [[nodiscard]] HWY_INLINE static HWY_LANES_CONSTEXPR extent_type
            out_lanes() {
                return hn::Lanes(Dout());
            }
            [[nodiscard]] HWY_INLINE static HWY_LANES_CONSTEXPR extent_type
            input_align() {
                return unroll() * in_lanes() * factor();
            }
            [[nodiscard]] HWY_INLINE static HWY_LANES_CONSTEXPR extent_type
            output_align() {
                return unroll() * out_lanes();
            }

            explicit State(const std::shared_ptr<const Layer> layer) : m_layer(std::move(layer)) {
                HWY_ASSERT(input_size() == output_size());
                HWY_ASSERT(input_chunks() / factor() == output_chunks());
            }

            [[nodiscard]] HWY_INLINE extent_type
            input_size() const {
                return m_layer->input_size();
            }
            [[nodiscard]] HWY_INLINE extent_type
            output_size() const {
                return m_layer->output_size();
            }
            [[nodiscard]] HWY_INLINE extent_type
            padded_input_size() const {
                return utils::pad_up(input_size(), input_align());
            }
            [[nodiscard]] HWY_INLINE extent_type
            padded_output_size() const {
                return utils::pad_up(output_size(), output_align());
            }
            [[nodiscard]] HWY_INLINE extent_type
            input_chunks() const {
                return padded_input_size() / (in_lanes() * unroll());
            }
            [[nodiscard]] HWY_INLINE extent_type
            output_chunks() const {
                return padded_input_size() / (out_lanes() * unroll());
            }

            [[nodiscard]] HWY_INLINE BufferConstraints<const input_type>
                                     input_buffer_constraints() const override {
                return {padded_input_size(), in_lanes() * sizeof(input_type)};
            }
            [[nodiscard]] HWY_INLINE BufferConstraints<output_type>
                                     output_buffer_constraints() const override {
                return {padded_output_size(), out_lanes() * sizeof(output_type)};
            }

            [[nodiscard]] HWY_INLINE auto
            input_span(const std::span<const input_type> span) const {
                using ext_t = extents<extent_type,
                                      std::dynamic_extent,
                                      unroll(),
                                      factor(),
                                      EXTENT_IF_LANES_CONSTEXPR(in_lanes())>;
                HWY_ASSERT(span.size() >= padded_input_size());
                auto* HWY_RESTRICT ptr = span.data();
                ext_t              ext{output_chunks(), unroll(), factor(), in_lanes()};
                return mdspan<const input_type, ext_t>{ptr, ext};
            }

            [[nodiscard]] HWY_INLINE auto
            output_span(const std::span<output_type> span) const {
                using ext_t =
                    std::extents<extent_type, std::dynamic_extent, unroll(), EXTENT_IF_LANES_CONSTEXPR(out_lanes())>;
                HWY_ASSERT(span.size() >= padded_output_size());
                auto* HWY_RESTRICT ptr = span.data();
                ext_t              ext{output_chunks(), unroll(), out_lanes()};
                return mdspan<output_type, ext_t>{ptr, ext};
            }

            template <size_t N, typename Tag, typename GetRegFn>
            [[nodiscard]] HWY_INLINE static auto
            ordered_demote_tree(GetRegFn&& get_reg) {
                if constexpr (N == 1) {
                    return get_reg(0);
                } else {
                    constexpr size_t half = N / 2;
                    auto             get  = [&](const size_t i) {
                        return hn::OrderedDemote2To(hn::RepartitionToNarrow<Tag>(), get_reg(2 * i), get_reg(2 * i + 1));
                    };
                    return ordered_demote_tree<half, hn::RepartitionToNarrow<Tag>>(get);
                }
            }

            void
            forward(std::span<const input_type> input, std::span<output_type> output) const override {
                const auto in  = input_span(input);
                const auto out = output_span(output);

                for (extent_type c = 0; c < in.extent(0); ++c) {
                    nnue::HWY_NAMESPACE::RegisterBank<unroll() * factor(), Vin>::run(
                        [&](const size_t i) { return hn::Load(Din(), &in[c, i / factor(), i % factor(), 0]); },
                        [&](auto get_reg, auto set_reg) {
                            if constexpr (Layer::shift() != 0) {
                                for (extent_type r = 0; r < unroll() * factor(); ++r) {
                                    set_reg(r, hn::ShiftRight<Layer::shift()>(get_reg(r)));
                                }
                            }
                            for (extent_type u = 0; u < in.extent(1); ++u) {
                                Vout v_out = ordered_demote_tree<factor(), Din>(
                                    [&](const size_t i) { return get_reg(u * factor() + i); });
                                v_out = hn::Max(hn::Zero(Dout()), v_out);
                                hn::Store(v_out, Dout(), &out[c, u, 0]);
                            }
                        });
                }
            }

          private:
            const std::shared_ptr<const Layer> m_layer;
        };

        CHEPP_AFTER_LAYER()
    }; // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::relu

HWY_AFTER_NAMESPACE();

#endif // CHEPP_RELU_INL_H
