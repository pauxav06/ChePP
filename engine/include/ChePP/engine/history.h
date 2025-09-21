#ifndef HISTORY_H
#define HISTORY_H

#include "movegen.h"
#include "position.h"

#include <functional>
#include <cassert>

using BonusT = int;

template <typename F>
concept BonusFn = requires(F f, BonusT x) {
    { f(x) } -> std::convertible_to<BonusT>;
};


struct HistoryTable
{
    [[nodiscard]] BonusT get_bonus(const Position& position, const Move move) const
    {
        return m_hist.at(position.piece_at(move.from_sq())).at(move.to_sq());
    }

    template <BonusFn F>
    void apply_bonus(const Position& position, const Move move, const F& bonus)
    {
        m_hist.at(position.piece_at(move.from_sq())).at(move.to_sq()) =
            bonus(m_hist.at(position.piece_at(move.from_sq())).at(move.to_sq()));
    }

    template <BonusFn F>
    void decay(const Position& position, const MoveList& moves, const F& decay)
    {
        for (const auto move : moves)
        {
            m_hist.at(position.piece_at(move.from_sq())).at(move.to_sq()) =
                decay(m_hist.at(position.piece_at(move.from_sq())).at(move.to_sq()));
        }
    }

    EnumArray<Piece, EnumArray<Square, int>> m_hist;
};


struct ContinuationHistoryTable
{
    HistoryTable& get_relevant_history(const Position& position)
    {
        const Move move = position.move();
        return m_hist.at(position.piece_at(move.to_sq())).at(move.to_sq());
    }

    const HistoryTable& get_relevant_history(const Position& position) const
    {
        const Move move = position.move();
        return m_hist.at(position.piece_at(move.to_sq())).at(move.to_sq());
    }

    [[nodiscard]] BonusT get_bonus(const Position& previous, const Position& current, const Move move) const
    {
        return get_relevant_history(previous).get_bonus(current, move);
    }

    template <BonusFn F>
    void apply_bonus(const Position& previous, const Position& current, const Move move, const F& bonus)
    {
        get_relevant_history(previous).apply_bonus(current, move, bonus);
    }

    template <BonusFn F>
    void decay(const Position& previous, const Position& current, const MoveList& moves, const F& decay)
    {
        get_relevant_history(previous).decay(current, moves, decay);
    }

    EnumArray<Piece, EnumArray<Square, HistoryTable>> m_hist;
};


struct History
{
    History() = default;
    History(HistoryTable* hist, Positions::Handle pos) : m_hist(hist), position(pos) {}

    History(const History&) = default;
    History& operator=(const History&) = default;
    History(History&&) noexcept = default;
    History& operator=(History&&) noexcept = default;

    ~History() = default;

    [[nodiscard]] BonusT get_bonus(const Move move) const
    {
        return m_hist->get_bonus(*position, move);
    }

    template <BonusFn F>
    void apply_bonus(const Move move, const F& bonus) {
        m_hist->apply_bonus(*position, move, bonus);
    }

    template <BonusFn F>
    void decay(const MoveList& moves, const F& decay)
    {
        m_hist->decay(*position, moves, decay);
    }

    HistoryTable* m_hist = nullptr;
    Positions::Handle position{};
};


struct ContinuationHistory
{
    using Bonus = std::function<int(int)>;
    using Decay = std::function<int(int)>;

    ContinuationHistory() = default;
    ContinuationHistory(ContinuationHistoryTable* hist, const Positions::Handle& prev) : m_hist(hist), previous(prev) {}

    ContinuationHistory(const ContinuationHistory&) = default;
    ContinuationHistory& operator=(const ContinuationHistory&) = default;
    ContinuationHistory(ContinuationHistory&&) noexcept = default;
    ContinuationHistory& operator=(ContinuationHistory&&) noexcept = default;

    ~ContinuationHistory() = default;

    [[nodiscard]] BonusT get_bonus(const Position& current, const Move move) const
    {
        if (previous->move() == Move::null()) return 0;
        return m_hist->get_bonus(*previous, current, move);
    }

    template <BonusFn F>
    void apply_bonus(const Position& current, const Move move, const F& bonus)
    {
        if (previous->move() == Move::null()) return;
        m_hist->apply_bonus(*previous, current, move, bonus);
    }

    template <BonusFn F>
    void decay(const Position& current, const MoveList& moves, const F& decay)
    {
        if (previous->move() == Move::null()) return;
        m_hist->decay(*previous, current, moves, decay);
    }


    ContinuationHistoryTable* m_hist = nullptr;
    Positions::Handle previous{};
};

using RefutationMapT = std::unordered_map<Move, size_t>;


#endif // HISTORY_H
