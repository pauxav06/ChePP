#ifndef SIMPLE_NNUE_H
#define SIMPLE_NNUE_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core.h"
#include "nnue.h"
#include "weights.h"

#include <hwy/base.h>
#include <iomanip>

namespace chepp::nnue {
    struct FeatureTransformer {
        static constexpr auto MaxChanges = 32;
        using FeatureT                   = uint16_t;
        using RetT                       = ArrayStack<FeatureT, MaxChanges>;

        static constexpr size_t n_features_v = 32 * 11 * 64;

        static bool
        needs_refresh(const Position& prev, const Position& cur, const Color view) {
            return prev.ksq(view) != cur.ksq(view);
        }

        static RetT
        get_refresh_features(const Position& pos, const Color side) {
            RetT res{};
            for (const Square sq : pos.occupancy()) {
                res.push_back(get_index(side, pos.ksq(side), sq, pos.piece_at(sq)));
            }
            return res;
        }

        static std::pair<RetT, RetT>
        get_incremental_features(const Position& prev, const Position& cur, const Color side) {
            std::pair<RetT, RetT> res{};
            for (const auto color : Color::all()) {
                for (const auto sq : prev.occupancy(color) ^ cur.occupancy(color)) {
                    if (prev.occupancy(color).is_set(sq)) {
                        res.second.push_back(get_index(side, prev.ksq(side), sq, prev.piece_at(sq)));
                    } else {
                        res.first.push_back(get_index(side, cur.ksq(side), sq, cur.piece_at(sq)));
                    }
                }
            }
            return res;
        }

      private:
        static int
        king_square_index(Square ksq) {
            static EnumArray<int, Square> WKSqH = {0,  1,  2,  3,  3,  2,  1,  0,  4,  5,  6,  7,  7,  6,  5,  4,
                                                   8,  9,  10, 11, 11, 10, 9,  8,  12, 13, 14, 15, 15, 14, 13, 12,
                                                   16, 17, 18, 19, 19, 18, 17, 16, 20, 21, 22, 23, 23, 22, 21, 20,
                                                   24, 25, 26, 27, 27, 26, 25, 24, 28, 29, 30, 31, 31, 30, 29, 28};

            return WKSqH[ksq];
        }

        static int
        get_index(Color view, Square king_square, Square piece_square, Piece piece) {
            auto relative_piece_square = (view == WHITE ? piece_square : piece_square.flipped_horizontally());
            auto relative_king_square  = (view == WHITE ? king_square : king_square.flipped_horizontally());
            if (king_square.file() > FILE_D) {
                relative_piece_square = relative_piece_square.flipped_vertically();
            }
            int piece_idx = piece.type() == KING ? 0 : 1 + piece.type().value() * 2 + (piece.color() == view ? 0 : 1);
            return king_square_index(relative_king_square) + relative_piece_square.value() * 32 + piece_idx * 32 * 64;
        }
    };

    struct Arch : std::enable_shared_from_this<Arch> {

        static constexpr std::size_t buckets = 8;
        using ft                             = FeatureTransformer;
        using accum_t                        = AccumulatorLayer<uint16_t, ft::n_features_v, int16_t, 1024>;
        using psqt_t                         = AccumulatorLayer<uint16_t, ft::n_features_v, int32_t, buckets>;
        using act0_t                         = ClippedReLULayer<int16_t, 1024, uint8_t, 1>;
        using l1_t                           = AffineLayer<uint8_t, 2048, int32_t, 16, int8_t, int32_t>;
        using act1_t                         = ClippedReLULayer<int32_t, 16, uint8_t, 64>;
        using l2_t                           = AffineLayer<uint8_t, 16, int32_t, 32, int8_t, int32_t>;
        using act2_t                         = ClippedReLULayer<int32_t, 32, uint8_t, 64>;
        using l3_t                           = AffineLayer<uint8_t, 32, int32_t, 1, int8_t, int32_t>;

        using all_layers_t = std::tuple<accum_t, psqt_t, act0_t, l1_t, act1_t, l2_t, act2_t, l3_t>;

        inline static KernelRegistry&
        registry() {
            static KernelRegistry registry{};
            static std::once_flag once;
            std::call_once(once, register_all_layers, registry);
            return registry;
        }

        struct Layers {
            std::shared_ptr<accum_t>                   accum{std::make_shared<accum_t>(acc_weights[0], acc_biases[0])};
            std::shared_ptr<psqt_t>                    psqt{std::make_shared<psqt_t>(psqt_weights[0], psqt_biases[0])};
            std::shared_ptr<act0_t>                    act0{std::make_shared<act0_t>()};
            std::array<std::shared_ptr<l1_t>, buckets> l1{
                create_array<buckets>([](auto i) { return std::make_shared<l1_t>(l1_weights[i], l1_biases[i]); })};
            std::shared_ptr<act1_t>                    act1{std::make_shared<act1_t>()};
            std::array<std::shared_ptr<l2_t>, buckets> l2{
                create_array<buckets>([](auto i) { return std::make_shared<l2_t>(l2_weights[i], l2_biases[i]); })};
            std::shared_ptr<act2_t>                    act2{std::make_shared<act2_t>()};
            std::array<std::shared_ptr<l3_t>, buckets> l3{
                create_array<buckets>([](auto i) { return std::make_shared<l3_t>(out_weights[i], out_biases[i]); })};
        };

        inline static Layers&
        layers() {
            static Layers layers{};
            return layers;
        }

        struct Kernels {
            std::shared_ptr<accum_t::IKernel> accum{};
            std::shared_ptr<psqt_t::IKernel>  psqt{};

            std::shared_ptr<act0_t::IKernel>                    act0{};
            std::array<std::shared_ptr<l1_t::IKernel>, buckets> l1{};
            std::shared_ptr<act1_t::IKernel>                    act1{};
            std::array<std::shared_ptr<l2_t::IKernel>, buckets> l2{};
            std::shared_ptr<act2_t::IKernel>                    act2{};
            std::array<std::shared_ptr<l3_t::IKernel>, buckets> l3{};
        };

        struct Memory {
            explicit Memory(const Kernels& kernels)
                : m_accum_stack(
                      {MAX_PLY + 1,
                       2,
                       accum_t::output_size() + std::max(kernels.accum->padding(), kernels.act0->input_padding())}),
                  m_psqt_stack({MAX_PLY + 1, 2, psqt_t::output_size() + kernels.psqt->padding()}) {
                std::size_t act0_padding = std::max(kernels.act0->output_padding(), kernels.l1[0]->input_padding());
                act0.resize(act0_t::size() * 2 + act0_padding);
                std::size_t l1_padding = std::max(kernels.l1[0]->output_padding(), kernels.act1->input_padding());
                l1.resize(l1_t::output_size() + l1_padding);
                std::size_t act1_padding = std::max(kernels.act1->output_padding(), kernels.l2[0]->input_padding());
                act1.resize(act1_t::size() + act1_padding);
                std::size_t l2_padding = std::max(kernels.l2[0]->output_padding(), kernels.act2->input_padding());
                l2.resize(l2_t::output_size() + l2_padding);
                std::size_t act2_padding = std::max(kernels.act2->output_padding(), kernels.l3[0]->input_padding());
                act2.resize(act2_t::size() + act2_padding);
                std::size_t l3_padding = kernels.l3[0]->output_padding();
                l3.resize(l3_t::output_size() + l3_padding);
            }

            void
            push() {
                ++m_idx;
            }
            void
            pop() {
                --m_idx;
            }

            auto
            accum(const Color color) {
                return m_accum_stack[{m_idx, color.value()}];
            }
            auto
            prev_accum(const Color color) {
                return m_accum_stack[{m_idx - 1, color.value()}];
            }
            auto
            psqt(const Color color) {
                return m_psqt_stack[{m_idx, color.value()}];
            }
            auto
            prev_psqt(const Color color) {
                return m_psqt_stack[{m_idx - 1, color.value()}];
            }

            hwy::AlignedVector<act0_t::output_type> act0{};
            hwy::AlignedVector<l1_t::output_type>   l1{};
            hwy::AlignedVector<act1_t::output_type> act1{};
            hwy::AlignedVector<l2_t::output_type>   l2{};
            hwy::AlignedVector<act2_t::output_type> act2{};
            hwy::AlignedVector<l2_t::output_type>   l3{};

          private:
            std::size_t                                 m_idx{};
            hwy::AlignedNDArray<accum_t::value_type, 3> m_accum_stack;
            hwy::AlignedNDArray<psqt_t::value_type, 3>  m_psqt_stack;
        };

        struct Network {
            explicit Network(const std::shared_ptr<Kernels>& kernels)
                : m_kernels(kernels), m_memory(std::make_shared<Memory>(*m_kernels)) {
            }

            [[nodiscard]] auto
            bucket() const {
                return m_buckets.back();
            }

            int32_t
            forward(const Color side) {
                const auto [pos, psqt] = forward_bucket(side, bucket());
                return pos + psqt;
            }

            void
            dbg_uci(const Color side) {
                std::cout << "\n================ NNUE Debug (UCI) ================\n";
                std::cout << "Side to move: " << (side == WHITE ? "WHITE" : "BLACK") << "\n";
                std::cout << "Active bucket: " << bucket() << "\n\n";

                std::cout << std::setw(8) << "Bucket" << std::setw(12) << "PSQT" << std::setw(12) << "Pos"
                          << std::setw(12) << "Total"
                          << "   Active\n";

                std::cout << "---------------------------------------------------\n";

                for (std::size_t b = 0; b < buckets; ++b) {
                    const auto [pos, psqt] = forward_bucket(side, b);
                    auto total             = pos + psqt;

                    std::cout << std::setw(8) << b << std::setw(12) << psqt << std::setw(12) << pos << std::setw(12)
                              << total << "   " << (b == bucket() ? "<-- ACTIVE" : "") << "\n";
                }

                std::cout << "===================================================\n\n";
            }

            void
            init(const Position& pos) {
                m_buckets.push_back((pos.occupancy().popcount() - 1) / 4);
                m_memory->push();
                for (const Color side : Color::all()) {
                    refresh(pos, side);
                }
            }

            void
            update(const Position& prev, const Position& next) {
                m_buckets.push_back((next.occupancy().popcount() - 1) / 4);
                m_memory->push();
                for (const Color side : Color::all()) {
                    if (ft::needs_refresh(prev, next, side)) {
                        refresh(next, side);
                    } else {
                        update(prev, next, side);
                    }
                }
            }

            void
            undo() {
                m_buckets.pop_back();
                m_memory->pop();
            }

          private:
            std::pair<int32_t, int32_t>
            forward_bucket(const Color side, const std::size_t bucket) {
                // TODO could lead to alignement problems if layer size is odd, maybe keep separate buffers?
                HWY_DASSERT(hwy::IsAligned(m_memory->act0.data() + accum_t::output_size()));

                m_kernels->act0->forward(m_memory->accum(side).data(), m_memory->act0.data());
                m_kernels->act0->forward(m_memory->accum(~side).data(), m_memory->act0.data() + accum_t::output_size());

                m_kernels->l1[bucket]->forward(m_memory->act0.data(), m_memory->l1.data());
                m_kernels->act1->forward(m_memory->l1.data(), m_memory->act1.data());
                m_kernels->l2[bucket]->forward(m_memory->act1.data(), m_memory->l2.data());
                m_kernels->act2->forward(m_memory->l2.data(), m_memory->act2.data());
                m_kernels->l3[bucket]->forward(m_memory->act2.data(), m_memory->l3.data());

                auto out = m_memory->l3[0];
                out *= 600;
                out /= (64 * 127);

                const auto psqt = (m_memory->psqt(side)[bucket] - m_memory->psqt(~side)[bucket]) / 2;

                return {out, psqt};
            }
            void
            refresh(const Position& pos, const Color side) {
                const auto features = ft::get_refresh_features(pos, side);
                m_kernels->accum->forward(features.data(), features.size(), m_memory->accum(side).data());
                m_kernels->psqt->forward(features.data(), features.size(), m_memory->psqt(side).data());
            }

            void
            update(const Position& prev, const Position& next, const Color side) {
                const auto [added_features, removed_features] = ft::get_incremental_features(prev, next, side);
                m_kernels->accum->forward_incremental(m_memory->prev_accum(side).data(),
                                                      added_features.data(),
                                                      added_features.size(),
                                                      removed_features.data(),
                                                      removed_features.size(),
                                                      m_memory->accum(side).data());
                m_kernels->psqt->forward_incremental(m_memory->prev_psqt(side).data(),
                                                     added_features.data(),
                                                     added_features.size(),
                                                     removed_features.data(),
                                                     removed_features.size(),
                                                     m_memory->psqt(side).data());
            }

            std::shared_ptr<Kernels> m_kernels;
            std::shared_ptr<Memory>  m_memory;
            std::vector<std::size_t> m_buckets;
        };

        inline static Network
        make_network(const std::shared_ptr<Kernels>& kernels) {
            return Network(kernels);
        }

        inline static std::shared_ptr<Kernels>
        make_tuned_kernels(const std::stop_token& stop_token) {
            return std::make_shared<Kernels>(registry().get_best_kernel(layers().accum, stop_token),
                                             registry().get_best_kernel(layers().psqt, stop_token),
                                             registry().get_best_kernel(layers().act0, stop_token),
                                             create_array<buckets>([&](auto i) {
                                                 return registry().get_best_kernel(layers().l1.at(i), stop_token);
                                             }),
                                             registry().get_best_kernel(layers().act1, stop_token),
                                             create_array<buckets>([&](auto i) {
                                                 return registry().get_best_kernel(layers().l2.at(i), stop_token);
                                             }),
                                             registry().get_best_kernel(layers().act2, stop_token),
                                             create_array<buckets>([&](auto i) {
                                                 return registry().get_best_kernel(layers().l3.at(i), stop_token);
                                             }));
        }

        inline static std::shared_ptr<Kernels>
        make_default_kernels() {
            return std::make_shared<Kernels>(
                registry().make_kernel(layers().accum, HWY_STATIC_TARGET, default_config).value(),
                registry().make_kernel(layers().psqt, HWY_STATIC_TARGET, default_config).value(),
                registry().make_kernel(layers().act0, HWY_STATIC_TARGET, default_config).value(),
                create_array<buckets>([](auto i) {
                    return registry().make_kernel(layers().l1.at(i), HWY_STATIC_TARGET, default_config).value();
                }),
                registry().make_kernel(layers().act1, HWY_STATIC_TARGET, default_config).value(),
                create_array<buckets>([](auto i) {
                    return registry().make_kernel(layers().l2.at(i), HWY_STATIC_TARGET, default_config).value();
                }),
                registry().make_kernel(layers().act2, HWY_STATIC_TARGET, default_config).value(),
                create_array<buckets>([](auto i) {
                    return registry().make_kernel(layers().l3.at(i), HWY_STATIC_TARGET, default_config).value();
                }));
        }
    };
} // namespace chepp::nnue

#endif