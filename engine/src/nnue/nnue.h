#ifndef CHEPP_LAYERS_H
#define CHEPP_LAYERS_H

#include "accumulator.h"
#include "affine.h"
#include "feature_transformer.h"
#include "layer_base.h"
#include "relu.h"

#include <hwy/targets.h>

#include <utility>

#include "weights/weights.h"

namespace chepp::nnue {
    template <typename Operation>
    struct Layer {
        using layer_t   = std::shared_ptr<typename Operation::layer_t>;
        using ikernel_t = std::shared_ptr<typename Operation::ikernel_t>;

        template <std::input_or_output_iterator It>
        static auto
        make_layer(It& begin, const It& end) {
            return Operation::make_layer(begin, end);
        }

        static auto
        make_kernel(const KernelRegistry& r, const layer_t& layer) {
            return r.make_kernel<Operation>(layer, HWY_STATIC_TARGET, default_config);
        }

        static auto
        make_best_kernel(const KernelRegistry& r, const layer_t& layer, const std::stop_token& s) {
            return r.make_best_kernel<Operation>(layer, s);
        }
    };

    template <typename Node, std::size_t N>
    struct Bucket {
        using layer_t   = std::array<typename Node::layer_t, N>;
        using ikernel_t = std::array<typename Node::ikernel_t, N>;

        template <std::input_or_output_iterator It>
        static auto
        make_layer(It& begin, const It& end) {
            return make_array<N>([&](auto) { return Node::make_layer(begin, end); });
        }

        static auto
        make_kernel(const KernelRegistry& r, const layer_t& layer) {
            return make_array<N>([&](auto i) { return Node::make_kernel(r, layer.at(i)); });
        }

        static auto
        make_best_kernel(const KernelRegistry& r, const layer_t& layer, const std::stop_token& s) {
            return make_array<N>([&](auto i) { return Node::make_best_kernel(r, layer.at(i), s); });
        }
    };

    template <typename... Nodes>
    struct Multiple {
        using layer_t   = std::tuple<typename Nodes::layer_t...>;
        using ikernel_t = std::tuple<typename Nodes::ikernel_t...>;

        template <std::size_t I>
        using node_t = std::tuple_element_t<I, std::tuple<Nodes...>>;

        static constexpr std::size_t N = sizeof...(Nodes);

        template <std::input_or_output_iterator It>
        static auto
        make_layer(It& begin, const It& end) {
            return make_tuple<N>([&](auto i) { return node_t<i>::make_layer(begin, end); });
        }

        static auto
        make_best_kernel(const KernelRegistry& r, const layer_t& l, const std::stop_token& s) {
            return make_tuple<N>([&](auto i) { return node_t<i>::make_best_kernel(r, std::get<i>(l), s); });
        }

        static auto
        make_kernel(const KernelRegistry& r, const layer_t& l) {
            return make_tuple<N>([&](auto i) { return node_t<i>::make_kernel(r, std::get<i>(l)); });
        }
    };

    template <typename Layer, auto... Configs>
    struct LayerConfig {
        using layer_t                 = Layer;
        static constexpr auto configs = std::make_tuple(Configs...);
    };

    struct Arch {
        using ft = FeatureTransformer;

        static constexpr std::size_t buckets = 8;

        using accum_t = Accumulator<uint16_t, ft::n_features_v, int16_t, 1024>;
        using psqt_t  = Accumulator<uint16_t, ft::n_features_v, int32_t, buckets>;
        using relu0_t = ClippedRelu<int16_t, 1024, uint8_t, 1>;
        using aff0_t  = Affine<uint8_t, 2048, int32_t, 16, int8_t, int32_t>;
        using relu1_t = ClippedRelu<int32_t, 16, uint8_t, 64>;
        using aff1_t  = Affine<uint8_t, 16, int32_t, 32, int8_t, int32_t>;
        using relu2_t = ClippedRelu<int32_t, 32, uint8_t, 64>;
        using aff2_t  = Affine<uint8_t, 32, int32_t, 1, int8_t, int32_t>;

        using Topology = Multiple<Layer<accum_t>,
                                  Layer<psqt_t>,
                                  Layer<relu0_t>,
                                  Bucket<Layer<aff0_t>, buckets>,
                                  Layer<relu1_t>,
                                  Bucket<Layer<aff1_t>, buckets>,
                                  Layer<relu2_t>,
                                  Bucket<Layer<aff2_t>, buckets>>;

        using layer_t   = Topology::layer_t;
        using ikernel_t = Topology::ikernel_t;

        struct MemSizes {
            std::size_t acc{}, psqt{}, relu0{}, aff0{}, relu1{}, aff1{}, relu2{}, aff2{};
        };

        static MemSizes
        make_mem_sizes(const ikernel_t& kernels) {
            MemSizes res{};
            const auto& [acc, psqt, relu0, aff0, relu1, aff1, relu2, aff2] = kernels;
            res.acc   = accum_t::output_size_v + std::max(acc->padding(), relu0->input_padding());
            res.psqt  = psqt_t::output_size_v + psqt->padding();
            res.relu0 = relu0_t::size_v * 2 +
                        std::max(relu0->output_padding(),
                                 std::ranges::max(aff0 | std::views::transform(&aff0_t::ikernel_t::input_padding)));
            res.aff0 = aff1_t::output_size_v +
                       std::max(std::ranges::max(aff0 | std::views::transform(&aff0_t::ikernel_t::output_padding)),
                                relu1->input_padding());
            res.relu1 = relu1_t::size_v +
                        std::max(relu1->output_padding(),
                                 std::ranges::max(aff1 | std::views::transform(&aff1_t::ikernel_t::input_padding)));
            res.aff1 = aff1_t::output_size_v +
                       std::max(std::ranges::max(aff1 | std::views::transform(&aff1_t::ikernel_t::output_padding)),
                                relu2->input_padding());
            res.relu2 = relu2_t::size_v +
                        std::max(relu2->output_padding(),
                                 std::ranges::max(aff2 | std::views::transform(&aff2_t::ikernel_t::input_padding)));
            res.aff2 = aff2_t::output_size_v +
                       std::ranges::max(aff2 | std::views::transform(&aff2_t::ikernel_t::output_padding));
            return res;
        }

        struct Memory {
            explicit Memory(const MemSizes& mem) noexcept
                : m_accum_stack({MAX_PLY, 2, mem.acc}), m_psqt_stack({MAX_PLY, 2, mem.psqt}),
                  m_relu(std::max({mem.relu0, mem.relu1, mem.relu2})), m_aff(std::max({mem.aff0, mem.aff1, mem.aff2})) {
            }

            Memory(Memory&&) = default;
            Memory&
            operator=(Memory&&) = default;

            void
            push() noexcept {
                ++m_top;
            }
            void
            pop() noexcept {
                --m_top;
            }
            auto
            accum(Color c) noexcept {
                return m_accum_stack[{m_top, c.index()}];
            }
            auto
            psqt(Color c) noexcept {
                return m_psqt_stack[{m_top, c.index()}];
            }
            auto
            prev_accum(Color c) noexcept {
                return m_accum_stack[{m_top - 1, c.index()}];
            }
            auto
            prev_psqt(Color c) noexcept {
                return m_psqt_stack[{m_top - 1, c.index()}];
            }
            auto
            relu() noexcept {
                return std::span{m_relu};
            }
            auto
            aff() noexcept {
                return std::span{m_aff};
            }

          private:
            std::size_t                     m_top{};
            hwy::AlignedNDArray<int16_t, 3> m_accum_stack;
            hwy::AlignedNDArray<int32_t, 3> m_psqt_stack;
            hwy::AlignedVector<uint8_t>     m_relu;
            hwy::AlignedVector<int32_t>     m_aff;
        };

        struct Network {
            explicit Network(const ikernel_t& kernels) noexcept
                : m_kernels(kernels), m_memory(make_mem_sizes(kernels)) {
            }

            Network(Network&&) = default;
            Network&
            operator=(Network&&) = default;

            [[nodiscard]] auto
            bucket() const noexcept {
                return m_buckets.back();
            }

            int32_t
            forward(const Color side) noexcept {
                const auto [pos, psqt] = forward_bucket(side, bucket());
                return pos + psqt;
            }

            void
            dbg_uci(const Color side) noexcept {
                fmt::print(std::cout, "{:<8}{:<8}{:<8}{}\n", "Bucket", "PSQT", "Pos", "Total");

                for (std::size_t b = 0; b < buckets; ++b) {
                    const auto [pos, psqt] = forward_bucket(side, b);
                    auto total             = pos + psqt;

                    fmt::print(std::cout, "{:<8}{:<8}{:<8}{}{}\n", b, psqt, pos, total, (b == bucket() ? " <-" : ""));
                }
            }

            void
            init(const Position& pos) noexcept {
                m_buckets.push_back(static_cast<size_t>((pos.occupancy().popcount() - 1) / 4));
                m_memory.push();
                for (const Color side : Color::all()) {
                    refresh(pos, side);
                }
            }

            void
            update(const Position& prev, const Position& next) noexcept {
                m_buckets.push_back(static_cast<size_t>((next.occupancy().popcount() - 1) / 4));
                m_memory.push();
                for (const Color side : Color::all()) {
                    if (ft::needs_refresh(prev, next, side)) {
                        refresh(next, side);
                    } else {
                        update(prev, next, side);
                    }
                }
            }

            void
            undo() noexcept {
                m_buckets.pop_back();
                m_memory.pop();
            }

          private:
            std::pair<int32_t, int32_t>
            forward_bucket(const Color side, const std::size_t bucket) noexcept {
                /* TODO could lead to alignement problems if layer size is odd
                   Fix by implementing a bi-affine layer taking in 2 input pointers and only applying biase once */
                HWY_DASSERT(hwy::IsAligned(m_memory.aff().data() + accum_t::output_size_v));

                const auto& [accum, psqt, act0, l1, act1, l2, act2, l3] = m_kernels;

                act0->forward(m_memory.accum(side).data(), m_memory.relu().data());
                act0->forward(m_memory.accum(~side).data(), m_memory.relu().data() + accum_t::output_size_v);

                l1[bucket]->forward(m_memory.relu().data(), m_memory.aff().data());
                act1->forward(m_memory.aff().data(), m_memory.relu().data());
                l2[bucket]->forward(m_memory.relu().data(), m_memory.aff().data());
                act2->forward(m_memory.aff().data(), m_memory.relu().data());
                l3[bucket]->forward(m_memory.relu().data(), m_memory.aff().data());

                auto out = m_memory.aff()[0];
                out *= 600;
                out /= (64 * 127);

                const auto mat = (m_memory.psqt(side)[bucket] - m_memory.psqt(~side)[bucket]) / 2;

                return {out, mat};
            }
            void
            refresh(const Position& pos, const Color side) noexcept {
                const auto features                                     = ft::get_refresh_features(pos, side);
                const auto& [accum, psqt, act0, l1, act1, l2, act2, l3] = m_kernels;
                accum->forward(features.data(), features.size(), m_memory.accum(side).data());
                psqt->forward(features.data(), features.size(), m_memory.psqt(side).data());
            }

            void
            update(const Position& prev, const Position& next, const Color side) noexcept {
                const auto [added_features, removed_features] = ft::get_incremental_features(prev, next, side);
                const auto& [accum, psqt, act0, l1, act1, l2, act2, l3] = m_kernels;

                accum->forward_incremental(m_memory.prev_accum(side).data(),
                                           added_features.data(),
                                           added_features.size(),
                                           removed_features.data(),
                                           removed_features.size(),
                                           m_memory.accum(side).data());
                psqt->forward_incremental(m_memory.prev_psqt(side).data(),
                                          added_features.data(),
                                          added_features.size(),
                                          removed_features.data(),
                                          removed_features.size(),
                                          m_memory.psqt(side).data());
            }

            std::vector<std::size_t> m_buckets;
            ikernel_t                m_kernels;
            Memory                   m_memory;
        };

        static auto
        make_layers() {
            auto       begin = std::ranges::begin(GENERATED_WEIGHTS);
            const auto end   = std::ranges::end(GENERATED_WEIGHTS);
            auto       res   = Topology::make_layer(begin, end);
            HWY_ASSERT(std::distance(begin, end) == 0);
            return res;
        }

        static auto
        make_kernels(const KernelRegistry& r, const layer_t& l) {
            return Topology::make_kernel(r, l);
        }

        static auto
        make_best_kernels(const KernelRegistry& r, const layer_t& l, const std::stop_token& s) {
            return Topology::make_best_kernel(r, l, s);
        }

        // clangd-format off
        using layers = std::tuple<LayerConfig<accum_t,
                                              default_config,
                                              AccumulatorSimd{1},
                                              AccumulatorSimd{2},
                                              AccumulatorSimd{4},
                                              AccumulatorSimd{8},
                                              AccumulatorSimd{16}>,
                                  LayerConfig<psqt_t, default_config>,
                                  LayerConfig<relu0_t,
                                              default_config,
                                              ClippedReluSimd{1},
                                              ClippedReluSimd{2},
                                              ClippedReluSimd{4},
                                              ClippedReluSimd{8},
                                              ClippedReluSimd{16}>,
                                  LayerConfig<aff0_t,
                                              default_config,
                                              AffineSimdRowMaj{1, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{2, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{4, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{8, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdRowMaj{1, AffineOperation::MulPairwiseAdd},
                                              AffineSimdRowMaj{2, AffineOperation::MulPairwiseAdd},
                                              AffineSimdRowMaj{4, AffineOperation::MulPairwiseAdd},
                                              AffineSimdRowMaj{8, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>,
                                  LayerConfig<relu1_t, default_config>,
                                  LayerConfig<aff1_t,
                                              default_config,
                                              AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd},
                                              AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd},
                                              AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>,
                                  LayerConfig<relu2_t, default_config>,
                                  LayerConfig<aff2_t, default_config>>;
        // clangd-format on
    };

    // This variable controls which layers will be compiled
    static constexpr auto ALL_LAYERS = std::tuple_cat(Arch::layers{});

    void
    register_all_layers(KernelRegistry&);
} // namespace chepp::nnue

#endif // CHEPP_LAYERS_H
