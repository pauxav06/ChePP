#include <hwy/aligned_allocator.h>
#include <hwy/auto_tune.h>

#include <experimental/mdarray>
#include <experimental/mdspan>
#include <iostream>
#include <mutex>

#include "affine.h"
#include "layer_cache.h"
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
        namespace nn = chepp::nnue::HWY_NAMESPACE;

        using namespace std::experimental;
        using namespace hwy;
        using namespace meta;

        template <typename Params>
        struct Layer;

        template <typename Types>
        concept I8I32Concept =
            std::is_same_v<typename Types::in, int8_t> && std::is_same_v<typename Types::out, int32_t>;

        template <KernelConcept Kernel, TypesConcept Types>
        struct Layer<std::tuple<Kernel, Types>> {
            using extent_type = size_t;
            using input_type  = Types::in;
            using output_type = Types::out;

            using weights_extent_t = std::extents<extent_type, std::dynamic_extent, std::dynamic_extent>;
            using biases_extent_t  = std::extents<extent_type, std::dynamic_extent>;

            using weights_span_t = std::mdspan<input_type, weights_extent_t>;
            using biases_span_t  = std::mdspan<output_type, biases_extent_t>;

            using state_t = AffineState<input_type, output_type>;

            const extent_type              m_input_size;
            const extent_type              m_output_size;
            const int                      m_bucket;
            const std::shared_ptr<state_t> m_state;
            const weights_span_t           m_weights_span;
            const biases_span_t            m_biases_span;

            explicit Layer(AffineParams params)
                : m_input_size(params.dims.in), m_output_size(params.dims.out), m_bucket(params.bucket),
                  m_state(g_cache.get_or_create(
                      m_input_size ^ m_output_size ^ m_bucket,
                      [&]() {
                          state_t state(m_input_size, m_output_size);
                          std::memcpy(state.weights.data(), params.weights.data(), params.weights.size_bytes());
                          std::memcpy(state.biases.data(), params.biases.data(), params.biases.size_bytes());
                          return state;
                      })),
                  m_weights_span(m_state.weights.data(), m_input_size, m_output_size),
                  m_biases_span(m_state.biases.data(), m_output_size) {
                HWY_ASSERT(params.weights.size() == m_input_size * m_output_size);
                HWY_ASSERT(params.biases.size() == m_output_size);
                HWY_ASSERT(m_state);
            }

            void init_state(AffineState<>)

                void forward(std::span<const input_type> input_view, std::span<output_type> output_view) {
                HWY_ASSERT(input_view.size() == m_input_size);
                HWY_ASSERT(output_view.size() == m_output_size);

                const auto* HWY_RESTRICT input_ptr  = input_view.data();
                auto* HWY_RESTRICT       output_ptr = output_view.data();

                for (extent_type r = 0; r < m_weights_span.extent(0); ++r) {
                    output_type acc = 0;
                    for (extent_type c = 0; c < m_weights_span.extent(1); ++c)
                        acc += static_cast<output_type>(m_weights_span[r, c]) * static_cast<output_type>(input_ptr[c]);
                    output_ptr[r] = acc + m_biases_span[r];
                }
            }
        };

        /**

        template <TypesConcept Types, DimsConcept Dims, UnrollConcept U, OperationConcept Op>
            requires (I8I32Concept<Types> && Dims::in % 4 == 0)
        struct Layer<std::tuple<SimdColMaj, Types, Dims, U, Op>> {
            using extent_type       = std::size_t;
            using input_type        = Types::in;
            using output_type       = Types::out;

            CONSTEXPR_EXTENT(input_size, Dims::in)
            CONSTEXPR_EXTENT(output_size, Dims::out)
            CONSTEXPR_EXTENT(unroll, U::value)
            CONSTEXPR_EXTENT(pack, 4)
            CONSTEXPR_EXTENT(padded_input_size, utils::pad_up(input_size * 1, pack * unroll))
            CONSTEXPR_EXTENT(input_padding, padded_input_size -  input_size)
            CONSTEXPR_EXTENT(input_valid, pack * unroll - input_padding)

            using Di8    = hn::CappedTag<input_type, input_size * 2>;
            using Du8    = hn::RebindToUnsigned<Di8>;
            using Di16   = hn::RepartitionToWide<Di8>;
            using Di32   = hn::RepartitionToWide<Di16>;
            using Veci8  = hn::VFromD<Di8>;
            using Vecu8  = hn::VFromD<Du8>;
            using Veci16 = hn::VFromD<Di16>;
            using Veci32 = hn::VFromD<Di32>;



            CONSTEXPR_EXTENT(chunks, padded_input_size / (pack * unroll))
            MAYBE_CONSTEXPR_EXTENT(d8_lanes, hn::Lanes(Di8()))
            MAYBE_CONSTEXPR_EXTENT(d32_lanes, hn::Lanes(Di32()))
            MAYBE_CONSTEXPR_EXTENT(padded_output_size, utils::pad_up(1 * output_size, 1 * d32_lanes))
            MAYBE_CONSTEXPR_EXTENT(output_padding, padded_output_size - output_size)
            MAYBE_CONSTEXPR_EXTENT(blocks, padded_output_size / d32_lanes)

            inline static HWY_LANES_CONSTEXPR auto s_tiled_weights =
                utils::make_tensor<input_type>(blocks, chunks, unroll, d8_lanes);
            inline static HWY_LANES_CONSTEXPR auto s_tiled_biases  = utils::make_tensor<output_type>(blocks, d32_lanes);
            inline static HWY_LANES_CONSTEXPR auto s_packed_input  = utils::make_tensor<output_type>(chunks, unroll);
            inline static HWY_LANES_CONSTEXPR auto s_padded_output = utils::make_tensor<output_type>(blocks, d32_lanes);

            inline static decltype(s_tiled_weights)::array_type s_tiled_weights_array;
            inline static decltype(s_tiled_biases)::array_type  s_tiled_biases_array;
            inline static std::once_flag                        s_init_flag;

            static void load_weights(std::span<const input_type> w_view, std::span<const output_type> b_view) {
                HWY_ASSERT(w_view.size() == input_size * output_size);
                HWY_ASSERT(b_view.size() == output_size);

                std::call_once(s_init_flag, [&]() {
                    s_tiled_weights_array = s_tiled_weights.make_array(s_tiled_weights.extent());
                    s_tiled_biases_array  = s_tiled_biases.make_array(s_tiled_biases.extent());

                    using namespace matrix;
                    const MatrixView w{w_view.data(), output_size, input_size};
                    const auto       w_t0 = pad(w, output_padding, input_padding);
                    const auto       w_t1 = tile_cols(w_t0, pack);
                    const auto       w_t2 = tile_cols(w_t1, d8_lanes);
                    materialize(w_t2, s_tiled_weights_array.data());

                    const MatrixView b{b_view.data(), output_size, 1};
                    const auto       b_t0 = pad(b, output_padding, 0);
                    materialize(b_t0, s_tiled_biases_array.data());
                });
            }

            template <extent_type N, typename GetFunc>
            static HWY_INLINE Veci32 reduce_tree(GetFunc&& get) {
                if constexpr (N == 1) {
                    return get(0);
                } else {
                    constexpr extent_type Half  = N / 2;
                    const Veci32          left  = reduce_tree<Half>([&](const size_t i) { return get(i); });
                    const Veci32          right = reduce_tree<N - Half>([&](const size_t i) { return get(i + Half); });
                    return left + right;
                }
            }

            static constexpr auto dot = [](const Vecu8 a, const Veci8 b, const Veci32 acc) {
                if constexpr (std::is_same_v<Op, SumOfMulQuadAcc>) {
                    return hn::SumOfMulQuadAccumulate(Di32(), a, b, acc);
                } else if constexpr (std::is_same_v<Op, SumOfMulPairAcc>) {
                    const Veci16 sum0 = hn::SatWidenMulPairwiseAdd(Di16(), a, b);
                    const Veci32 sum1 = hn::WidenMulPairwiseAdd(Di32(), sum0, hn::Set(Di16(), 1));
                    return hn::Add(acc, sum1);
                } else {
                    static_assert(false, "No int8*int8->int32 operation was found");
                    return hn::Undefined(Di32());
                }
            };

            static void forward(std::span<const input_type> input_view, std::span<output_type> output_view) {
                thread_local auto tmp_out = s_padded_output.make_array(s_padded_output.extent());

                HWY_ASSERT(input_view.size() == input_size);
                HWY_ASSERT(output_view.size() == output_size);

                auto* HWY_RESTRICT input_ptr        = input_view.data();
                auto* HWY_RESTRICT output_ptr       = output_view.data();
                auto* HWY_RESTRICT packed_input_ptr = reinterpret_cast<const output_type * HWY_RESTRICT>(input_ptr);

                const auto packed_input = s_packed_input.make_const_span(packed_input_ptr, s_packed_input.extent());

                for (extent_type b = 0; b < s_tiled_weights_array.extent(0); ++b) {
                    nn::RegisterBank<unroll, Veci32>::run(
                        [&](const size_t u) {
                            return u == 0 ? hn::Load(Di32(), &s_tiled_biases_array[b, 0]) : hn::Zero(Di32());
                        },
                        [&](auto get_reg, auto set_reg) {
                            for (extent_type c = 0; c < s_tiled_weights_array.extent(1); ++c) {
                                for (extent_type u = 0; u < s_tiled_weights_array.extent(2); ++u) {
                                    set_reg(u,
                                            dot(hn::BitCast(Du8(), hn::Set(Di32(), packed_input[c, u])),
                                                hn::Load(Di8(), &s_tiled_weights_array[b, c, u, 0]),
                                                get_reg(u)));
                                }
                            }
                            const Veci32 out = reduce_tree<unroll>([&](size_t i) { return get_reg(i); });
                            hn::Store(out, Di32(), &tmp_out[b, 0]);
                        });
                    std::memcpy(output_ptr, tmp_out.data(), output_view.size_bytes());
                }
            }
        };
        **/
    }; // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::affine

HWY_AFTER_NAMESPACE();

#endif