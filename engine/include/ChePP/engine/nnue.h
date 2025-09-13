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

#include "network_net.h"
#include "position.h"

template <typename T, size_t MaxSize>
class ArrayStack
{
    std::array<T, MaxSize> data{};
    size_t                 topIndex = 0;

  public:
    bool empty() const { return topIndex == 0; }
    bool full() const { return topIndex == MaxSize; }

    bool push_back(const T& value)
    {
        if (full())
            return false;
        data[topIndex++] = value;
        return true;
    }

    bool pop()
    {
        if (empty())
            return false;
        --topIndex;
        return true;
    }

    T& top()
    {
        if (empty())
            throw std::underflow_error("Stack empty");
        return data[topIndex - 1];
    }

    const T& top() const
    {
        if (empty())
            throw std::underflow_error("Stack empty");
        return data[topIndex - 1];
    }

    size_t size() const { return topIndex; }

    auto begin() { return data.begin(); }
    auto end() { return data.begin() + topIndex; }

    auto begin() const { return data.begin(); }
    auto end() const { return data.begin() + topIndex; }
};

struct FeatureTransformer
{
    static constexpr auto MaxChanges = 32;
    using FeatureT                   = uint16_t;
    using RetT                       = ArrayStack<FeatureT, MaxChanges>;

    static constexpr size_t n_features_v = 16 * 12 * 64;

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
            cur.occupancy().for_each_square([&](const Square& sq)
                                            { add_v.push_back(get_index(view, cur.ksq(view), sq, cur.piece_at(sq))); });
        }
        else
        {
            auto add = [&](const Square sq, const Piece pc)
            { add_v.push_back(get_index(view, cur.ksq(view), sq, pc)); };
            auto rem = [&](const Square sq, const Piece pc)
            { rem_v.push_back(get_index(view, cur.ksq(view), sq, pc)); };

            const EnumArray<Color, Bitboard> color_diff = {
                prev.occupancy(WHITE) ^ cur.occupancy(WHITE),
                prev.occupancy(BLACK) ^ cur.occupancy(BLACK),
            };

            for (const auto c : {WHITE, BLACK})
            {
                color_diff.at(c).for_each_square( [&](const Square& sq){
                    if (prev.occupancy(c).is_set(sq)) rem(sq, prev.piece_at(sq));
                    else add(sq, cur.piece_at(sq));
                });
            }
        }
        return {add_v, rem_v};
    }

  private:
    static int king_square_index(int relative_king_square)
    {
        constexpr int indices[64]{
            -1, -1, -1, -1, 14, 14, 15, 15, //
            -1, -1, -1, -1, 14, 14, 15, 15, //
            -1, -1, -1, -1, 12, 12, 13, 13, //
            -1, -1, -1, -1, 12, 12, 13, 13, //
            -1, -1, -1, -1, 8,  9,  10, 11, //
            -1, -1, -1, -1, 8,  9,  10, 11, //
            -1, -1, -1, -1, 4,  5,  6,  7,  //
            -1, -1, -1, -1, 0,  1,  2,  3,  //
        };

        return indices[relative_king_square];
    }

    static int get_index(Color view, Square king_sq, Square piece_sq, Piece piece)
    {

        const PieceType piece_type  = piece.type();
        const Color     piece_color = piece.color();

        int piece_square = piece_sq.value() ^ 56;
        int king_square  = king_sq.value() ^ 56;

        const int oP  = piece_type.value() + 6 * (piece_color != view);
        const int oK  = (7 * !(king_square & 4)) ^ (56 * view.value()) ^ king_square;
        const int oSq = (7 * !(king_square & 4)) ^ (56 * view.value()) ^ piece_square;

        return king_square_index(oK) * 12 * 64 + oP * 64 + oSq;
    }
};

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
using namespace hwy::HWY_NAMESPACE;

struct Accumulator
{
    static constexpr auto OutSz = 1024;
    static constexpr auto L1Sz  = 16;
    static constexpr auto L2Sz  = 32;
    using AccumulatorT          = std::array<int16_t, OutSz>;

  private:
    HWY_ALIGN AccumulatorT white_accumulator{};
    HWY_ALIGN AccumulatorT black_accumulator{};

  public:
    Accumulator() = default;
    explicit Accumulator(const Position& pos)
    {
        const auto [wadd, wrem] = FeatureTransformer::get_features(pos, pos, WHITE, true);
        refresh_acc(WHITE, wadd);
        const auto [badd, brem] = FeatureTransformer::get_features(pos, pos, BLACK, true);
        refresh_acc(BLACK, badd);
    }

    explicit Accumulator(const Accumulator& acc_prev, const Position& pos_cur, const Position& pos_prev)
    {
        update(acc_prev, pos_cur, pos_prev, WHITE);
        update(acc_prev, pos_cur, pos_prev, BLACK);
    }

    template <size_t UNROLL = 4>
    [[nodiscard]] int32_t evaluate(const Color view) const
    {
        const auto our_acc_ptr = static_cast<const int16_t*>(
            HWY_ASSUME_ALIGNED(view == WHITE ? white_accumulator.data() : black_accumulator.data(), 64));
        const auto their_acc_ptr = static_cast<const int16_t*>(
            HWY_ASSUME_ALIGNED(view == WHITE ? black_accumulator.data() : white_accumulator.data(), 64));

        using D32 = ScalableTag<int32_t>;
        using D16 = ScalableTag<int16_t>;

        using HalfD16 = FixedTag<int16_t, Lanes(D32{})>;
        using HalfD8  = FixedTag<int8_t, Lanes(D16{})>;


        HWY_ALIGN std::array<int32_t, L1Sz> l1_out{};
        std::memcpy(l1_out.data(), g_l1_biases, sizeof(g_l1_biases));
        HWY_ALIGN std::array<int32_t, L2Sz> l2_out{};
        std::memcpy(l2_out.data(), g_l2_biases, sizeof(g_l2_biases));
        HWY_ALIGN int32_t out = g_out_bias[0];

        // std::ranges::for_each(white_accumulator | std::views::take(10), [] (int16_t i) { std::cout << i << std::endl;
        // });


        for (size_t j = 0; j < OutSz; j += Lanes(D16{}) * UNROLL) {
            std::array<Vec<D16>, UNROLL> v_our_block{};
            std::array<Vec<D16>, UNROLL> v_their_block{};
            for (size_t u = 0; u < UNROLL; ++u) {
                const size_t idx = j + u * Lanes(HalfD8{});
                v_our_block[u]   = Min(Max(Load(D16{}, &our_acc_ptr[idx]), Zero(D16{})), Set(D16{}, 127*32));
                v_their_block[u] = Min(Max(Load(D16{}, &their_acc_ptr[idx]), Zero(D16{})), Set(D16{}, 127*32));
            }


            for (int i = 0; i < L1Sz; ++i) {
                Vec<D32> acc = Zero(D32{});

                for (size_t u = 0; u < UNROLL; ++u) {
                    const size_t idx = j + u * Lanes(D16{});
                    const Vec<D16> w_our   = Load(D16{}, &g_l1_weights[i * OutSz * 2 + idx]);
                    const Vec<D16> w_their = Load(D16{}, &g_l1_weights[i * OutSz * 2 + idx + OutSz]);

                    acc = Add(acc, WidenMulPairwiseAdd(D32{}, v_our_block[u], w_our));
                    acc = Add(acc, WidenMulPairwiseAdd(D32{}, v_their_block[u], w_their));
                }

                l1_out[i] += ReduceSum(D32{}, acc);
            }
        }

        // we get better precision by dividing in the end
        using QuantVec = CappedTag<int32_t, L1Sz>;
        for (int i = 0; i < L1Sz; i += Lanes(QuantVec{}))
        {
            Vec<D32> acc = Load(QuantVec{}, &l1_out[i]);
            acc = ShiftRight<16>(acc);
            Store(acc, QuantVec{}, &l1_out[i]);
        }


        // std::ranges::for_each(l1_out | std::views::take(10), [] (int16_t i) { std::cout << i << std::endl; });

        for (int i = 0; i < L2Sz; ++i)
        {
            HWY_ALIGN Vec<D32> acc = Zero(D32{});
            for (size_t j = 0; j < L1Sz; j += Lanes(HalfD16{}))
            {
                const Vec<D32> v = Max(Load(D32{}, &l1_out[j]), Zero(D32{}));
                const Vec<D32> w = PromoteTo(D32{}, Load(HalfD16{}, &g_l2_weights[i * L1Sz + j]));
                acc = Add(acc, Mul(v, w));
            }
            l2_out[i] += ReduceSum(D32{}, acc);
        }


        HWY_ALIGN Vec<D32> acc = Zero(D32{});
        for (size_t j = 0; j < L2Sz; j += Lanes(HalfD16{}))
        {
            const Vec<D32> v = Max(Load(D32{}, &l2_out[j]), Zero(D32{}));
            const Vec<D32> w = PromoteTo(D32{}, Load(HalfD16{}, &g_out_weights[j]));
            acc = Add(acc, Mul(v, w));
        }
        out += ReduceSum(D32{}, acc);

        return out >> 16;
    }

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

    template <size_t UNROLL = 8>
    void refresh_acc(const Color view, const FeatureTransformer::RetT& features)
    {
        auto& acc = (view == WHITE ? white_accumulator : black_accumulator);

        std::memcpy(acc.data(), g_ft_biases, OutSz * sizeof(int16_t));

        using D                         = ScalableTag<int16_t>;
        alignas(64) auto v_accumulators = std::array<decltype(Load(D{}, acc.data())), UNROLL>{};

        auto* HWY_RESTRICT acc_ptr = static_cast<int16_t*>(HWY_ASSUME_ALIGNED(acc.data(), 64));

        for (size_t i = 0; i < OutSz; i += UNROLL * Lanes(D{}))
        {
            for (size_t u = 0; u < UNROLL; ++u)
            {
                if (i + u * Lanes(D{}) < OutSz)
                    v_accumulators[u] = Load(D{}, &acc_ptr[i + u * Lanes(D{})]);
            }

            for (const auto f : features)
            {
                for (size_t u = 0; u < UNROLL; ++u)
                {
                    if (i + u * Lanes(D{}) < OutSz)
                    {
                        auto v_weights    = Load(D{}, &g_ft_weights[f * OutSz + i + u * Lanes(D{})]);
                        v_accumulators[u] = Add(v_accumulators[u], v_weights);
                    }
                }
            }

            for (size_t u = 0; u < UNROLL; ++u)
            {
                if (i + u * Lanes(D{}) < OutSz)
                    Store(v_accumulators[u], D{}, &acc_ptr[i + u * Lanes(D{})]);
            }
        }
    }

    template <size_t UNROLL = 8>
    void update_acc(const Accumulator& previous, const Color view, const FeatureTransformer::RetT& add,
                    const FeatureTransformer::RetT& sub)
    {
        auto& acc  = (view == WHITE ? white_accumulator : black_accumulator);
        auto& prev = (view == WHITE ? previous.white_accumulator : previous.black_accumulator);

        std::memcpy(acc.data(), prev.data(), OutSz * sizeof(int16_t));

        using D                         = ScalableTag<int16_t>;
        alignas(64) auto v_accumulators = std::array<decltype(Load(D{}, acc.data())), UNROLL>{};

        auto* HWY_RESTRICT acc_ptr = static_cast<int16_t*>(HWY_ASSUME_ALIGNED(acc.data(), 64));

        for (size_t i = 0; i < OutSz; i += UNROLL * Lanes(D{}))
        {
            for (size_t u = 0; u < UNROLL; ++u)
            {
                if (i + u * Lanes(D{}) < OutSz)
                    v_accumulators[u] = Load(D{}, &acc_ptr[i + u * Lanes(D{})]);
            }

            for (const auto f : add)
            {
                for (size_t u = 0; u < UNROLL; ++u)
                {
                    if (i + u * Lanes(D{}) < OutSz)
                    {
                        auto v_weights    = Load(D{}, &g_ft_weights[f * OutSz + i + u * Lanes(D{})]);
                        v_accumulators[u] = Add(v_accumulators[u], v_weights);
                    }
                }
            }

            for (const auto f : sub)
            {
                for (size_t u = 0; u < UNROLL; ++u)
                {
                    if (i + u * Lanes(D{}) < OutSz)
                    {
                        auto v_weights    = Load(D{}, &g_ft_weights[f * OutSz + i + u * Lanes(D{})]);
                        v_accumulators[u] = Sub(v_accumulators[u], v_weights);
                    }
                }
            }

            for (size_t u = 0; u < UNROLL; ++u)
            {
                if (i + u * Lanes(D{}) < OutSz)
                    Store(v_accumulators[u], D{}, &acc_ptr[i + u * Lanes(D{})]);
            }
        }
    }
};

HWY_AFTER_NAMESPACE();

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

    void do_move(const Position& prev, const Position& next)
    {
        m_accumulators.emplace_back(m_accumulators.back(), next, prev);
    }

    void undo_move() { m_accumulators.pop_back(); }

  private:
    std::vector<Accumulator> m_accumulators{};
};

#endif