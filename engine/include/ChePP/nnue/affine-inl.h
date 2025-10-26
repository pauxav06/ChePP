#include <hwy/aligned_allocator.h>
#include <hwy/auto_tune.h>

#include <experimental/mdarray>
#include <experimental/mdspan>
#include <iostream>
#include <mutex>
#include <tuple>
#include <utility>

#include "ChePP/engine/layers.h"
#include "affine.h"
#include "matrix.h"
#include "utils.h"

#if defined(CHEPP_AFFINE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_AFFINE_INL_H_
#undef CHEPP_AFFINE_INL_H_
#else
#define CHEPP_AFFINE_INL_H_
#endif

#include <hwy/highway.h>

#include "utils-inl.h"

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::layers::affine {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;

        using namespace std::experimental;
        using namespace hwy;

        CHEPP_BEFORE_LAYER()

        template <typename Layer, KernelConcept Kernel>
        struct State<Layer, Kernel> : Layer::IState {
            using extent_type = std::size_t;
            using input_type  = Layer::input_type;
            using output_type = Layer::output_type;

            using weights_extent_t = std::extents<extent_type, std::dynamic_extent, std::dynamic_extent>;
            using biases_extent_t  = std::extents<extent_type, std::dynamic_extent>;

            using weights_span_t = mdspan<const input_type, weights_extent_t>;
            using biases_span_t  = mdspan<const output_type, biases_extent_t>;

            [[nodiscard]] HWY_INLINE BufferConstraints<const input_type>
                                     input_buffer_constraints() const override {
                return {m_layer->input_size(), sizeof(input_type)};
            }
            [[nodiscard]] HWY_INLINE BufferConstraints<output_type>
                                     output_buffer_constraints() const override {
                return {m_layer->output_size(), sizeof(output_type)};
            }

            explicit State(std::shared_ptr<const Layer> layer) : m_layer(std::move(layer)) {
            }
            explicit State(std::shared_ptr<const Layer> layer, Kernel) : State(std::move(layer)) {
            }

            void
            init() override {
                m_weights = weights_span_t{m_layer->weights().data(),
                                           weights_extent_t{m_layer->output_size(), m_layer->input_size()}};
                m_biases  = biases_span_t{m_layer->biases().data(), biases_extent_t{m_layer->output_size()}};
            }

            void
            forward(std::span<const input_type> input_span, std::span<output_type> output_span) const override {
                HWY_ASSERT(input_buffer_constraints().span_satisfies(input_span));
                HWY_ASSERT(output_buffer_constraints().span_satisfies(output_span));

                const auto* HWY_RESTRICT input_ptr  = input_span.data();
                auto* HWY_RESTRICT       output_ptr = output_span.data();

                for (extent_type r = 0; r < m_weights.extent(0); ++r) {
                    output_type acc = 0;
                    for (extent_type c = 0; c < m_weights.extent(1); ++c)
                        acc += static_cast<output_type>(m_weights[r, c]) * static_cast<output_type>(input_ptr[c]);
                    output_ptr[r] = acc + m_biases[r];
                }
            }

          private:
            const std::shared_ptr<const Layer> m_layer;
            weights_span_t                     m_weights;
            biases_span_t                      m_biases;
        };

        template <typename Layer, UnrollConcept U, OperationConcept Op>
            requires(std::is_same_v<typename Layer::input_type, int8_t> &&
                     std::is_same_v<typename Layer::output_type, int32_t>)
        struct State<Layer, SimdColMaj, U, Op> : Layer::IState {
            using extent_type = std::size_t;
            using input_type  = Layer::input_type;
            using output_type = Layer::output_type;

            [[nodiscard]] HWY_INLINE static constexpr extent_type
            unroll() {
                return U::value;
            }
            [[nodiscard]] HWY_INLINE static constexpr extent_type
            pack() {
                return 4;
            }

            using Di8    = hn::ScalableTag<input_type>;
            using Du8    = hn::RebindToUnsigned<Di8>;
            using Di16   = hn::RepartitionToWide<Di8>;
            using Di32   = hn::RepartitionToWide<Di16>;
            using Veci8  = hn::VFromD<Di8>;
            using Vecu8  = hn::VFromD<Du8>;
            using Veci16 = hn::VFromD<Di16>;
            using Veci32 = hn::VFromD<Di32>;

            [[nodiscard]] HWY_INLINE static HWY_LANES_CONSTEXPR extent_type
            d8_lanes() {
                return hn::Lanes(Di8());
            }
            [[nodiscard]] HWY_INLINE static HWY_LANES_CONSTEXPR extent_type
            d32_lanes() {
                return hn::Lanes(Di32());
            }

            explicit State(std::shared_ptr<const Layer> layer) : m_layer(std::move(layer)) {
            }
            explicit State(std::shared_ptr<const Layer> layer, SimdColMaj, U, Op) : State(std::move(layer)) {
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
                return utils::pad_up(input_size(), pack() * unroll());
            }
            [[nodiscard]] HWY_INLINE extent_type
            input_padding() const {
                return padded_input_size() - input_size();
            }
            [[nodiscard]] HWY_INLINE extent_type
            chunks() const {
                return padded_input_size() / (pack() * unroll());
            }
            [[nodiscard]] HWY_INLINE extent_type
            padded_output_size() const {
                return utils::pad_up(output_size(), d32_lanes());
            }
            [[nodiscard]] HWY_INLINE extent_type
            output_padding() const {
                return padded_output_size() - output_size();
            }
            [[nodiscard]] HWY_INLINE extent_type
            blocks() const {
                return padded_output_size() / d32_lanes();
            }

            using weights_ext_t   = std::extents<extent_type,
                                                 std::dynamic_extent,
                                                 std::dynamic_extent,
                                                 unroll(),
                                                 EXTENT_IF_LANES_CONSTEXPR(d8_lanes())>;
            using weights_array_t = mdarray<input_type, weights_ext_t, layout_right, AlignedVector<input_type>>;
            [[nodiscard]] auto
            make_weights_array() const {
                return weights_array_t{blocks(), chunks(), unroll(), d8_lanes()};
            }

            using biases_ext_t = std::extents<extent_type, std::dynamic_extent, EXTENT_IF_LANES_CONSTEXPR(d32_lanes())>;
            using biases_array_t = mdarray<output_type, biases_ext_t, layout_right, AlignedVector<output_type>>;
            [[nodiscard]] auto
            make_biases_array() const {
                return biases_array_t{blocks(), d32_lanes()};
            }

            [[nodiscard]] HWY_INLINE BufferConstraints<const input_type>
                                     input_buffer_constraints() const override {
                return {padded_input_size(), d8_lanes() * sizeof(hn::TFromD<Di8>)};
            }
            [[nodiscard]] HWY_INLINE BufferConstraints<output_type>
                                     output_buffer_constraints() const override {
                return {padded_output_size(), d32_lanes() * sizeof(hn::TFromD<Di32>)};
            }

            [[nodiscard]] HWY_INLINE auto
            make_packed_input_span(const std::span<const input_type> span) const {
                using packed_type = output_type;
                using ext_t       = std::extents<extent_type, std::dynamic_extent, unroll()>;
                using span_t      = mdspan<const packed_type, ext_t>;

                HWY_ASSERT(input_buffer_constraints().span_satisfies(span));

                auto* HWY_RESTRICT ptr        = span.data();
                auto* HWY_RESTRICT packed_ptr = reinterpret_cast<const packed_type * HWY_RESTRICT>(ptr);

                return span_t{packed_ptr, ext_t{chunks(), unroll()}};
            }
            [[nodiscard]] HWY_INLINE auto
            make_padded_output_span(const std::span<output_type> span) const {
                using ext_t  = std::extents<extent_type, std::dynamic_extent, EXTENT_IF_LANES_CONSTEXPR(d32_lanes())>;
                using span_t = mdspan<output_type, ext_t>;

                HWY_ASSERT(output_buffer_constraints().span_satisfies(span));

                auto* HWY_RESTRICT ptr = span.data();

                return span_t{ptr, ext_t{blocks(), d32_lanes()}};
            }

            [[nodiscard]] HWY_INLINE bool
            has_vector_size_changed() const {
                return d8_lanes() != m_weights.extent(3);
            }

            void
            init() override {
                const matrix::MatrixView w{m_layer->weights().data(), output_size(), input_size()};
                const auto               w_t0 = pad(w, output_padding(), input_padding());
                const auto               w_t1 = tile_cols(w_t0, pack());
                const auto               w_t2 = tile_cols(w_t1, d8_lanes());

                const matrix::MatrixView b{m_layer->biases().data(), output_size(), 1};
                const auto               b_t0 = pad(b, output_padding(), 0);

                m_weights = make_weights_array();
                m_biases  = make_biases_array();

                materialize(w_t2, m_weights.data());
                materialize(b_t0, m_biases.data());
            }

            void
            forward(std::span<const input_type> input_span, std::span<output_type> output_span) const override {
                HWY_ASSERT(!has_vector_size_changed());

                const auto packed_input  = make_packed_input_span(input_span);
                const auto padded_output = make_padded_output_span(output_span);

                for (extent_type b = 0; b < m_weights.extent(0); ++b) {
                    nnue::HWY_NAMESPACE::RegisterBank<unroll(), Veci32>::run(
                        [&](const size_t u) { return u == 0 ? hn::Load(Di32(), &m_biases[b, 0]) : hn::Zero(Di32()); },
                        [&](auto get_reg, auto set_reg) {
                            for (extent_type c = 0; c < m_weights.extent(1); ++c) {
                                for (extent_type u = 0; u < m_weights.extent(2); ++u) {
                                    set_reg(u,
                                            dot(hn::BitCast(Du8(), hn::Set(Di32(), packed_input[c, u])),
                                                hn::Load(Di8(), &m_weights[b, c, u, 0]),
                                                get_reg(u)));
                                }
                            }
                            const Veci32 out = reduce_tree<unroll()>([&](size_t i) { return get_reg(i); });
                            hn::Store(out, Di32(), &padded_output[b, 0]);
                        });
                }
            }

          private:
            template <extent_type N, typename GetFunc>
            [[nodiscard]] HWY_INLINE static Veci32
            reduce_tree(GetFunc&& get) {
                if constexpr (N == 1) {
                    return get(0);
                } else {
                    constexpr extent_type Half  = N / 2;
                    const Veci32          left  = reduce_tree<Half>([&](const size_t i) { return get(i); });
                    const Veci32          right = reduce_tree<N - Half>([&](const size_t i) { return get(i + Half); });
                    return left + right;
                }
            }

            [[nodiscard]] HWY_INLINE static Veci32
            dot(const Vecu8 a, const Veci8 b, const Veci32 acc) {
                if constexpr (std::is_same_v<Op, SumOfMulQuadAcc>) {
                    return hn::SumOfMulQuadAccumulate(Di32(), a, b, acc);
                } else if constexpr (std::is_same_v<Op, SumOfMulPairAcc>) {
                    const Veci16 sum0 = hn::SatWidenMulPairwiseAdd(Di16(), a, b);
                    const Veci32 sum1 = hn::WidenMulPairwiseAdd(Di32(), sum0, hn::Set(Di16(), 1));
                    return hn::Add(acc, sum1);
                } else {
                    static_assert(false);
                    return hn::Undefined(Di32());
                }
            };

            const std::shared_ptr<const Layer> m_layer;
            weights_array_t                    m_weights;
            biases_array_t                     m_biases;
        };

        CHEPP_AFTER_LAYER()
    }; // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::affine
HWY_AFTER_NAMESPACE();

#endif