#ifndef SIMPLE_NNUE_H
#define SIMPLE_NNUE_H

#include "layers.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../nnue/layers.h"
#include "position.h"

#include "../nnue/simd/operations.h"
#include "layers.h"

namespace nnue
{
    struct FeatureTransformer
    {
        static constexpr auto MaxChanges = 32;
        using FeatureT                   = uint16_t;
        using RetT                       = ArrayStack<FeatureT, MaxChanges>;

        static constexpr size_t n_features_v = 32 * 11 * 64;

        static bool needs_refresh(const Position& cur, const Position& prev, const Color view)
        {
            return prev.ksq(view) != cur.ksq(view);
        }

        static std::pair<RetT, RetT> get_features(const Position& cur, const Position& prev, const Color view,
                                                  const bool refresh)

        {
            RetT add_v;
            RetT rem_v;

            if (refresh)
            {
                for (const Square sq : cur.occupancy())
                {
                    add_v.push_back(get_index(view, cur.ksq(view), sq, cur.piece_at(sq)));
                }
            }
            else
            {
                auto add = [&](const Square sq, const Piece pc)
                { add_v.push_back(get_index(view, cur.ksq(view), sq, pc)); };
                auto rem = [&](const Square sq, const Piece pc)
                { rem_v.push_back(get_index(view, cur.ksq(view), sq, pc)); };

                const EnumArray<Bitboard, Color> color_diff = {
                    prev.occupancy(WHITE) ^ cur.occupancy(WHITE),
                    prev.occupancy(BLACK) ^ cur.occupancy(BLACK),
                };

                for (const auto c : {WHITE, BLACK})
                {
                    for (const Square sq : color_diff.at(c))
                    {
                        if (prev.occupancy(c).is_set(sq)) rem(sq, prev.piece_at(sq));
                        else add(sq, cur.piece_at(sq));
                    }
                }
            }
            return {add_v, rem_v};
        }

      private:
        static int king_square_index(Square ksq)
        {
            static EnumArray<int, Square> WKSqH = {0,  1,  2,  3,  3,  2,  1,  0,  4,  5,  6,  7,  7,  6,  5,  4,
                                                   8,  9,  10, 11, 11, 10, 9,  8,  12, 13, 14, 15, 15, 14, 13, 12,
                                                   16, 17, 18, 19, 19, 18, 17, 16, 20, 21, 22, 23, 23, 22, 21, 20,
                                                   24, 25, 26, 27, 27, 26, 25, 24, 28, 29, 30, 31, 31, 30, 29, 28};

            return WKSqH[ksq];
        }

        static int get_index(Color view, Square king_square, Square piece_square, Piece piece)
        {
            auto relative_piece_square = (view == WHITE ? piece_square : piece_square.flipped_horizontally());
            auto relative_king_square  = (view == WHITE ? king_square : king_square.flipped_horizontally());
            if (king_square.file() > FILE_D)
            {
                relative_piece_square = relative_piece_square.flipped_vertically();
            }
            int piece_idx = piece.type() == KING ? 0 : 1 + piece.type().value() * 2 + (piece.color() == view ? 0 : 1);
            return king_square_index(relative_king_square) + relative_piece_square.value() * 32 + piece_idx * 32 * 64;
        }
    };

    template <typename Arch>
    struct Network;

    struct Accumulator
    {
        static constexpr auto AccSz  = 1024;
        static constexpr auto PsqtSz = 8;

        using AccumulatorT = std::array<int16_t, AccSz>;
        using PsqtT        = std::array<int32_t, PsqtSz>;

        Accumulator() = default;
        explicit Accumulator(const Position& pos)
        {
            const auto [wadd, wrem] = FeatureTransformer::get_features(pos, pos, WHITE, true);
            refresh_acc(WHITE, wadd);
            const auto [badd, brem] = FeatureTransformer::get_features(pos, pos, BLACK, true);
            refresh_acc(BLACK, badd);
            m_bucket = (pos.occupancy().popcount() - 1) / 4;
        }

        explicit Accumulator(const Accumulator& acc_prev, const Position& pos_cur, const Position& pos_prev)
        {
            update(acc_prev, pos_cur, pos_prev, WHITE);
            update(acc_prev, pos_cur, pos_prev, BLACK);
            m_bucket = (pos_cur.occupancy().popcount() - 1) / 4;

        }

        static NOINLINE Accumulator bench_refresh(const Position& pos) { return Accumulator(pos); }

        [[nodiscard]] size_t bucket() const { return m_bucket; }

      private:
        void update(const Accumulator& prev, const Position& pos_cur, const Position& pos_prev, const Color view)
        {
            const bool needs_refresh = FeatureTransformer::needs_refresh(pos_cur, pos_prev, view);
            const auto [add, rem]    = FeatureTransformer::get_features(pos_cur, pos_prev, view, needs_refresh);
            if (needs_refresh)
                refresh_acc(view, add);
            else
                update_acc(prev, view, add, rem);
        }

        // horribly slow and memory bound function, how to make it faster?
        // we cannot tile weights because they are accessed sparsely, and doing one feature at a time
        // would force intermediate stores bc we don't have enough registers
        void refresh_acc(const Color view, const FeatureTransformer::RetT& features)
        {
            using namespace simd;
            using AccumTag = pick_tag_t<int16_t>;
            using Vec16    = register_type_t<AccumTag>;
            using PsqtTag  = pick_tag_t<int32_t, PsqtSz * sizeof(int32_t) * 8>;
            using Vec32    = register_type_t<PsqtTag>;

            auto acc  = reinterpret_cast<Vec16*>(m_accumulators.at(view).data());
            auto psqt = reinterpret_cast<Vec32*>(m_psqts.at(view).data());

            __builtin_prefetch(acc_biases, 0, 3);
            __builtin_prefetch(psqt_biases, 0, 1);

            const auto acc_bias = reinterpret_cast<const Vec16*>(&acc_biases[0][0]);

            static constexpr int elems_per_reg  = lane_count_v<AccumTag>;
            static constexpr int n_regs         = register_count_v<typename AccumTag::arch> / 2;
            static constexpr int elems_per_bloc = n_regs * elems_per_reg;
            static constexpr int n_blocks       = sizeof(m_accumulators[WHITE]) / sizeof(int16_t) / elems_per_bloc;

            for (int b = 0; b < n_blocks; ++b)
            {
                Vec16 accumulators[n_regs];

                for (int r = 0; r < n_regs; ++r)
                    accumulators[r] = acc_bias[b * n_regs + r];

                for (const auto f : features)
                {
                    const auto weights = &reinterpret_cast<const Vec16*>(&acc_weights[0][f * AccSz])[b * n_regs];
                    for (int r = 0; r < n_regs; ++r)
                        accumulators[r] = add<AccumTag>(weights[r], accumulators[r]);
                }

                for (int r = 0; r < n_regs; ++r)
                    acc[b * n_regs + r] = accumulators[r];
            }

            const auto* psqt_bias        = reinterpret_cast<const Vec32*>(psqt_biases);
            auto        psqt_accumulator = psqt_bias[0];
            for (const auto f : features)
            {
                const auto w     = reinterpret_cast<const Vec32*>(&psqt_weights[0][f * PsqtSz]);
                psqt_accumulator = add<PsqtTag>(psqt_accumulator, w[0]);
            }
            psqt[0] = psqt_accumulator;
        };

        void update_acc(const Accumulator& previous, const Color view, const FeatureTransformer::RetT& added_features,
                        const FeatureTransformer::RetT& removed_features)
        {
            using namespace simd;
            using AccumTag = pick_tag_t<int16_t>;
            using Vec16    = register_type_t<AccumTag>;
            using PsqtTag  = pick_tag_t<int32_t, 8 * 32>;
            using Vec32    = register_type_t<PsqtTag>;

            auto acc  = reinterpret_cast<Vec16*>(m_accumulators.at(view).data());
            auto psqt = reinterpret_cast<Vec32*>(m_psqts.at(view).data());

            __builtin_prefetch(acc_biases, 0, 3);
            __builtin_prefetch(psqt_biases, 0, 1);

            const auto acc_bias = reinterpret_cast<const Vec16*>(previous.m_accumulators.at(view).data());

            static constexpr int elems_per_reg  = lane_count_v<AccumTag>;
            static constexpr int n_regs         = register_count_v<typename AccumTag::arch> / 2;
            static constexpr int elems_per_bloc = n_regs * elems_per_reg;
            static constexpr int n_blocks       = sizeof(m_accumulators[WHITE]) / sizeof(int16_t) / elems_per_bloc;

            for (int b = 0; b < n_blocks; ++b)
            {
                Vec16 accumulators[n_regs];

                for (int r = 0; r < n_regs; ++r)
                    accumulators[r] = acc_bias[b * n_regs + r];

                for (const auto f : added_features)
                {
                    const auto weights = &reinterpret_cast<const Vec16*>(&acc_weights[0][f * AccSz])[b * n_regs];
                    for (int r = 0; r < n_regs; ++r)
                        accumulators[r] = add<AccumTag>(weights[r], accumulators[r]);
                }
                for (const auto f : removed_features)
                {
                    const auto weights = &reinterpret_cast<const Vec16*>(&acc_weights[0][f * AccSz])[b * n_regs];
                    for (int r = 0; r < n_regs; ++r)
                        accumulators[r] = sub<AccumTag>(accumulators[r], weights[r]);
                }

                for (int r = 0; r < n_regs; ++r)
                    acc[b * n_regs + r] = accumulators[r];
            }

            const auto* psqt_bias        = reinterpret_cast<const Vec32*>(previous.m_psqts.at(view).data());
            auto        psqt_accumulator = psqt_bias[0];
            for (const auto f : added_features)
            {
                const auto w     = reinterpret_cast<const Vec32*>(&psqt_weights[0][f * PsqtSz]);
                psqt_accumulator = add<PsqtTag>(psqt_accumulator, w[0]);
            }
            for (const auto f : removed_features)
            {
                const auto w     = reinterpret_cast<const Vec32*>(&psqt_weights[0][f * PsqtSz]);
                psqt_accumulator = sub<PsqtTag>(psqt_accumulator, w[0]);
            }
            psqt[0] = psqt_accumulator;
        }

    public:
        alignas(64) EnumArray<AccumulatorT, Color> m_accumulators{};
        alignas(64) EnumArray<PsqtT, Color> m_psqts{};
        size_t m_bucket{};
    };

    struct Accumulators
    {
        using Acc         = Accumulator;
        using ConstAcc    = const Acc;
        using AccRef      = Acc&;
        using ConstAccRef = ConstAcc&;

        explicit Accumulators(const Position& pos)
        {
            m_accumulators.reserve(MAX_PLY);
            m_accumulators.emplace_back(pos);
        }

        std::span<Acc>                    accumulators() { return m_accumulators; }
        [[nodiscard]] std::span<ConstAcc> accumulators() const { return m_accumulators; }

        AccRef                    last() { return m_accumulators.back(); }
        [[nodiscard]] ConstAccRef last() const { return m_accumulators.back(); }

        [[nodiscard]] const Accumulator& operator[](const size_t i) const { return m_accumulators[i]; }
        Accumulator&                     operator[](const size_t i) { return m_accumulators[i]; }

        [[nodiscard]] uint32_t ply() const { return static_cast<int>(m_accumulators.size() - 1); }

        void do_move(const Position& prev, const Position& next)
        {
            m_accumulators.emplace_back(m_accumulators.back(), next, prev);
        }

        void undo_move() { m_accumulators.pop_back(); }

        using Handle = VectorHandle<Accumulator>;

        Handle handle_to_last()
        {
            return VectorHandle{&m_accumulators, static_cast<unsigned>(m_accumulators.size() - 1)};
        }

      private:
        std::vector<Accumulator> m_accumulators{};
    };

    template <std::size_t Buckets_ = 8, std::size_t AccSz_ = 1024, std::size_t PsqtSz_ = 8, std::size_t L1Sz_ = 16,
              std::size_t L2Sz_ = 32>
    struct ArchTpl
    {
        static constexpr std::size_t Buckets = Buckets_;
        static constexpr std::size_t AccSz   = AccSz_;
        static constexpr std::size_t PsqtSz  = PsqtSz_;
        static constexpr std::size_t L1Sz    = L1Sz_;
        static constexpr std::size_t L2Sz    = L2Sz_;

        using layer1  = affine::AffineLayer<L1Sz, 2 * AccSz>;
        using l1_relu = relu::ClippedRelu32_8<L1Sz, 6>;

        using layer2  = affine::AffineLayer<L2Sz, L1Sz>;
        using l2_relu = relu::ClippedRelu32_8<L2Sz, 6>;

        using layer3  = affine::AffineLayer<1, L2Sz>;

        using ft_relu = relu::QuantizedClippedRelu16_8<AccSz>;
    };

    template <typename Arch>
    struct Network
    {
        using arch_t    = Arch;
        using layer1_t  = typename arch_t::layer1;
        using l1_relu_t = typename arch_t::l1_relu;
        using layer2_t  = typename arch_t::layer2;
        using l2_relu_t = typename arch_t::l2_relu;
        using layer3_t  = typename arch_t::layer3;
        using ft_relu_t = typename arch_t::ft_relu;

        static constexpr std::size_t Buckets = arch_t::Buckets;
        static constexpr std::size_t AccSz   = arch_t::AccSz;
        static constexpr std::size_t L1Sz    = arch_t::L1Sz;
        static constexpr std::size_t L2Sz    = arch_t::L2Sz;

        std::array<layer1_t, Buckets> m_l1{};
        std::array<layer2_t, Buckets> m_l2{};
        std::array<layer3_t, Buckets> m_l3{};

        Network() = default;

        template <typename L1WArr, typename L1BArr, typename L2WArr, typename L2BArr, typename OUTWArr,
                  typename OUTBArr>
        void load_weights(const L1WArr& l1_w, const L1BArr& l1_b, const L2WArr& l2_w, const L2BArr& l2_b,
                          const OUTWArr& out_w, const OUTBArr& out_b)
        {
            for (std::size_t b = 0; b < Buckets; ++b)
            {
                m_l1.at(b).load_weights(l1_w[b], l1_b[b]);
                m_l2.at(b).load_weights(l2_w[b], l2_b[b]);
                m_l3.at(b).load_weights(out_w[b], out_b[b]);
            }
        }

        [[nodiscard]] int32_t evaluate(const Accumulator& acc, Color view, std::size_t bucket) const
        {
            alignas(64) thread_local std::array<int8_t, 2 * AccSz> l1_in{};
            alignas(64) thread_local std::array<int32_t, L1Sz>     l1_out{};
            alignas(64) thread_local std::array<int8_t, L1Sz>      l2_in{};
            alignas(64) thread_local std::array<int32_t, L2Sz>     l2_out{};
            alignas(64) thread_local std::array<int8_t, L2Sz>      l3_in{};
            alignas(64) thread_local int32_t                       out{};

            ft_relu_t::forward(acc.m_accumulators.at(view).data(), l1_in.data());
            ft_relu_t::forward(acc.m_accumulators.at(~view).data(), l1_in.data() + AccSz);
            m_l1[bucket].forward(l1_in.data(), l1_out.data());
            l1_relu_t::forward(l1_out.data(), l2_in.data());
            m_l2[bucket].forward(l2_in.data(), l2_out.data());
            l2_relu_t::forward(l2_out.data(), l3_in.data());
            m_l3[bucket].forward(l3_in.data(), &out);

            out *= 600;
            out /= (64 * 127);

            const int32_t psqt = (acc.m_psqts[view][bucket] - acc.m_psqts[~view][bucket]) / 2;

            return out + psqt;
        }

        [[nodiscard]] int32_t evaluate(const Accumulator& acc, Color view) const
        {
            return evaluate(acc, view, acc.m_bucket);
        }

        void evaluate_uci(const Accumulator& acc, const Color view) const
        {
            for (std::size_t i = 0; i < Buckets; ++i)
            {
                std::cout << std::format("Eval for bucket {} : {}", i, evaluate(acc, view, i));
                if (i == acc.m_bucket)
                {
                    std::cout << " <- active bucket";
                }
                std::cout << std::endl;
            }
        }

        [[nodiscard]] NOINLINE int32_t bench_eval(const Accumulator& acc, Color view) const
        {
            return evaluate(acc, view);
        }
    };

    inline Network<ArchTpl<>> network;

    struct Initialiser
    {
        Initialiser()
        {
            network.load_weights(l1_weights, l1_biases, l2_weights, l2_biases, out_weights, out_biases);
        }
    };
} // namespace nnue

#endif