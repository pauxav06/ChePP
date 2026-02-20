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

#include "accumulator.h"
#include "affine.h"
#include "layers.h"
#include "position.h"
#include "relu.h"

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

    using namespace layers;

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

        inline static KernelRegistry registry{};

        static void
        register_kernels();
        static void
        init_kernels() {
            std::call_once(register_flag, [&] {
                register_kernels();
                accum_kernel = registry.get_best_kernel(accum_layer);
                psqt_kernel  = registry.get_best_kernel(psqt_layer);
                act0_kernel  = registry.get_best_kernel(act0_layer);
                l1_kernels   = create_array<buckets>([&](auto i) { return registry.get_best_kernel(l1_layers.at(i)); });
                act1_kernel  = registry.get_best_kernel(act1_layer);
                l2_kernels   = create_array<buckets>([&](auto i) { return registry.get_best_kernel(l2_layers.at(i)); });
                act2_kernel  = registry.get_best_kernel(act2_layer);
                l3_kernels   = create_array<buckets>([&](auto i) { return registry.get_best_kernel(l3_layer.at(i)); });
            });
        }

        struct Network {
            explicit Network()
                : m_accum_allocator(accum_t::output_size() +
                                    std::max(accum_kernel->padding(), act0_kernel->input_padding())),
                  m_psqt_allocator(psqt_t::output_size() + psqt_kernel->padding()),
                  m_act0_out(act0_t::size() * 2 +
                             std::max(act0_kernel->output_padding(), l1_kernels[0]->input_padding())),
                  m_l1_out(l1_t::output_size() +
                           std::max(l1_kernels[0]->output_padding(), act1_kernel->input_padding())),
                  m_act1_out(act1_t::size() + std::max(act1_kernel->output_padding(), l2_kernels[0]->input_padding())),
                  m_l2_out(l2_t::output_size() +
                           std::max(l2_kernels[0]->output_padding(), act2_kernel->input_padding())),
                  m_act2_out(act2_t::size() + std::max(act2_kernel->output_padding(), l3_kernels[0]->input_padding())),
                  m_out(l3_t::output_size() + l3_kernels[0]->output_padding()) {
            }

            int32_t
            forward(const Color side) {
                const auto [pos, psqt] = forward_bucket(side, m_bucket);
                return pos + psqt;
            }

            void
            dbg_uci(const Color side) {
                std::cout << "\n================ NNUE Debug (UCI) ================\n";
                std::cout << "Side to move: " << (side == WHITE ? "WHITE" : "BLACK") << "\n";
                std::cout << "Active bucket: " << m_bucket << "\n\n";

                std::cout << std::setw(8) << "Bucket" << std::setw(12) << "PSQT" << std::setw(12) << "Pos"
                          << std::setw(12) << "Total"
                          << "   Active\n";

                std::cout << "---------------------------------------------------\n";

                for (std::size_t b = 0; b < buckets; ++b) {
                    const auto [pos, psqt] = forward_bucket(side, b);
                    auto total             = pos + psqt;

                    std::cout << std::setw(8) << b << std::setw(12) << psqt << std::setw(12) << pos << std::setw(12)
                              << total << "   " << (b == m_bucket ? "<-- ACTIVE" : "") << "\n";
                }

                std::cout << "===================================================\n\n";
            }

            void
            init(const Position& pos) {
                m_bucket = (pos.occupancy().popcount() - 1) / 4;
                for (const Color side : Color::all()) {
                    m_accum_stack.at(side).resize(1);
                    m_psqt_stack.at(side).resize(1);
                    m_accum_stack.at(side).at(0) = m_accum_allocator.get();
                    m_psqt_stack.at(side).at(0)  = m_psqt_allocator.get();
                    refresh(pos, side);
                }
            }

            void
            update(const Position& prev, const Position& next) {
                m_bucket = (next.occupancy().popcount() - 1) / 4;
                for (const Color side : Color::all()) {
                    m_accum_stack.at(side).push_back(m_accum_allocator.get());
                    m_psqt_stack.at(side).push_back(m_psqt_allocator.get());
                    if (ft::needs_refresh(prev, next, side)) {
                        refresh(next, side);
                    } else {
                        update(prev, next, side);
                    }
                }
            }

            void
            undo() {
                for (const Color side : Color::all()) {
                    m_accum_stack.at(side).pop_back();
                    m_psqt_stack.at(side).pop_back();
                }
            }

            template <typename T>
            struct Alloc {
                static constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
                Alloc(std::size_t size)
                    : m_mem({MAX_PLY * 2 + Square::count(), size}), m_free(MAX_PLY * 2 + Square::count()) {
                    std::iota(m_free.begin(), m_free.end(), 0);
                }

                struct Handle {
                    Handle() noexcept = default;
                    Handle(Alloc* alloc, std::size_t idx) : m_alloc(alloc), m_idx(idx) {
                    }

                    Handle(const Handle& other) = delete;
                    Handle&
                    operator=(const Handle& other) = delete;

                    Handle(Handle&& other) noexcept
                        : m_alloc(std::exchange(other.m_alloc, nullptr)), m_idx(std::exchange(other.m_idx, none)) {
                    }

                    Handle&
                    operator=(Handle&& other) noexcept {
                        if (std::addressof(other) != this) {
                            m_alloc = std::exchange(other.m_alloc, nullptr);
                            m_idx   = std::exchange(other.m_idx, none);
                        }
                        return *this;
                    }

                    [[nodiscard]] explicit operator bool() {
                        return m_alloc != nullptr && m_idx != none;
                    }

                    [[nodiscard]] std::span<T>
                    mem() const {
                        auto res = m_alloc->m_mem[{m_idx}];
                        return {res.data(), res.size()};
                    }

                    ~Handle() {
                        if (*this) {
                            m_alloc->free(m_idx);
                        }
                    }

                  private:
                    Alloc*      m_alloc{nullptr};
                    std::size_t m_idx{none};
                };

                Handle
                get() {
                    if (const auto idx = allocate(); idx != none) {
                        return {this, idx};
                    }
                    return {};
                }

                void
                free(std::size_t idx) {
                    if (idx != none) {
                        m_free.push_back(idx);
                    }
                }

                std::size_t
                allocate() {
                    HWY_ASSERT(!m_free.empty());
                    auto res = m_free.back();
                    m_free.pop_back();
                    return res;
                }

              private:
                hwy::AlignedNDArray<T, 2> m_mem;
                std::vector<std::size_t>  m_free;
            };

          private:
            std::pair<int32_t, int32_t>
            forward_bucket(const Color side, const std::size_t bucket) {
                act0_kernel->forward(m_accum_stack.at(side).back().mem().data(), m_act0_out.data());
                act0_kernel->forward(m_accum_stack.at(~side).back().mem().data(),
                                     m_act0_out.data() + accum_t::output_size());
                // std::ranges::copy(m_act0_out | std::views::take(10), std::ostream_iterator<int>{std::cout, " "});

                l1_kernels.at(bucket)->forward(m_act0_out.data(), m_l1_out.data());
                act1_kernel->forward(m_l1_out.data(), m_act1_out.data());
                l2_kernels.at(bucket)->forward(m_act1_out.data(), m_l2_out.data());
                act2_kernel->forward(m_l2_out.data(), m_act2_out.data());
                l3_kernels.at(bucket)->forward(m_act2_out.data(), m_out.data());

                auto out = m_out[0];
                out *= 600;
                out /= (64 * 127);

                const int32_t psqt =
                    (m_psqt_stack.at(side).back().mem()[bucket] - m_psqt_stack.at(~side).back().mem()[bucket]) / 2;

                return {out, psqt};
            }
            void
            refresh(const Position& pos, const Color side) {
                const auto features = ft::get_refresh_features(pos, side);
                accum_kernel->forward(features.data(), features.size(), m_accum_stack.at(side).back().mem().data());
                psqt_kernel->forward(features.data(), features.size(), m_psqt_stack.at(side).back().mem().data());
            }

            void
            update(const Position& prev, const Position& next, const Color side) {
                const auto idx                                = m_psqt_stack.at(side).size() - 1;
                const auto [added_features, removed_features] = ft::get_incremental_features(prev, next, side);
                accum_kernel->forward_incremental(m_accum_stack.at(side).at(idx - 1).mem().data(),
                                                  added_features.data(),
                                                  added_features.size(),
                                                  removed_features.data(),
                                                  removed_features.size(),
                                                  m_accum_stack.at(side).at(idx).mem().data());
                psqt_kernel->forward_incremental(m_psqt_stack.at(side).at(idx - 1).mem().data(),
                                                 added_features.data(),
                                                 added_features.size(),
                                                 removed_features.data(),
                                                 removed_features.size(),
                                                 m_psqt_stack.at(side).at(idx).mem().data());
            }

            struct CacheEntry {
                Alloc<int16_t>::Handle accum_handle;
                Alloc<int32_t>::Handle psqt_handle;
                Position               position;
            };

            Alloc<int16_t>                                        m_accum_allocator;
            Alloc<int32_t>                                        m_psqt_allocator;
            EnumArray<std::vector<Alloc<int16_t>::Handle>, Color> m_accum_stack{};
            EnumArray<std::vector<Alloc<int32_t>::Handle>, Color> m_psqt_stack{};
            EnumArray<CacheEntry, Color, Square>                  m_cache{};
            EnumArray<std::vector<std::size_t>, Color, Square>    m_pos{};
            hwy::AlignedVector<uint8_t>                           m_act0_out{};
            hwy::AlignedVector<int32_t>                           m_l1_out{};
            hwy::AlignedVector<uint8_t>                           m_act1_out{};
            hwy::AlignedVector<int32_t>                           m_l2_out{};
            hwy::AlignedVector<uint8_t>                           m_act2_out{};
            hwy::AlignedVector<int32_t>                           m_out{};
            std::size_t                                           m_bucket{};
        };

        static Network
        make_network() {
            init_kernels();
            return Network{};
        }

        inline static std::shared_ptr<accum_t> accum_layer{std::make_shared<accum_t>(acc_weights[0], acc_biases[0])};
        inline static std::shared_ptr<accum_t::IKernel> accum_kernel;

        inline static std::shared_ptr<psqt_t> psqt_layer{std::make_shared<psqt_t>(psqt_weights[0], psqt_biases[0])};
        inline static std::shared_ptr<psqt_t::IKernel> psqt_kernel;

        inline static std::shared_ptr<act0_t>          act0_layer{std::make_shared<act0_t>()};
        inline static std::shared_ptr<act0_t::IKernel> act0_kernel;

        inline static std::array<std::shared_ptr<l1_t>, buckets> l1_layers{
            create_array<buckets>([](auto i) { return std::make_shared<l1_t>(l1_weights[i], l1_biases[i]); })};
        inline static std::array<std::shared_ptr<l1_t::IKernel>, buckets> l1_kernels;

        inline static std::shared_ptr<act1_t>          act1_layer{std::make_shared<act1_t>()};
        inline static std::shared_ptr<act1_t::IKernel> act1_kernel;

        inline static std::array<std::shared_ptr<l2_t>, buckets> l2_layers{
            create_array<buckets>([](auto i) { return std::make_shared<l2_t>(l2_weights[i], l2_biases[i]); })};
        inline static std::array<std::shared_ptr<l2_t::IKernel>, buckets> l2_kernels;

        inline static std::shared_ptr<act2_t>          act2_layer{std::make_shared<act2_t>()};
        inline static std::shared_ptr<act2_t::IKernel> act2_kernel;

        inline static std::array<std::shared_ptr<l3_t>, buckets> l3_layer{
            create_array<buckets>([](auto i) { return std::make_shared<l3_t>(out_weights[i], out_biases[i]); })};
        inline static std::array<std::shared_ptr<l3_t::IKernel>, buckets> l3_kernels;

      private:
        inline static std::once_flag register_flag;
    };

} // namespace chepp::nnue

#endif