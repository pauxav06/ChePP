#ifndef HISTORY_H
#define HISTORY_H

#include "position.h"

#include <cassert>
#include <functional>

template <typename T, T Range>
class Bonus {
    static_assert(std::is_arithmetic_v<T>, "Bonus requires a numeric type");

  public:
    Bonus() : value_(0) {}
    explicit Bonus(T v) : value_(v) {}

    operator T() const { return value_; }

    static constexpr T max() { return Range; }
    static constexpr T min() { return -Range; }

    Bonus operator<<(T newValue) {
        T clamped = std::clamp(newValue, -Range, Range);
        value_ += clamped - value_ * std::abs(clamped) / Range;
        return *this;
    }

    T    value() const { return value_; }
    void set(T v) { value_ = v; }

  private:
    T value_;
};

using HistoryBonus = Bonus<int, 16000>;

struct History {

    [[nodiscard]] HistoryBonus& at(const Position& position, const Move move) {
        assert(move.is_ok());
        return m_hist.at(position.piece_at(move.from_sq())).at(move.to_sq());
    }
    EnumArray<HistoryBonus, Piece, Square> m_hist;
};

struct ContinuationHistory {
    History& get_relevant_history(const Position& position) {
        const Move move = position.move();
        assert(move.is_ok());
        return m_hist.at(position.moved()).at(move.to_sq());
    }
    EnumArray<History, Piece, Square> m_hist;
};

struct CaptureHistory {
    [[nodiscard]] HistoryBonus& at(const Position& position, const Move move) {
        return m_hist.at(position.piece_at(move.from_sq())).at(move.to_sq()).at(position.captured_by_move(move));
    }
    EnumArray<HistoryBonus, Piece, Square, Piece> m_hist;
};

using RefutationHistory = std::unordered_map<Move, size_t>;

#endif // HISTORY_H
