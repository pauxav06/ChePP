//
// Created by paul on 9/11/25.
//

#ifndef HISTORY_H
#define HISTORY_H

#include "position.h"
#include <memory>
#include <vector>
#include <algorithm>

template <typename T>
using HistTableT = EnumArray<Piece, EnumArray<Square, T>>;

using HistTable = HistTableT<int>;

using ContHistTable = HistTableT<HistTableT<int>>;

struct HistoryManager {
    std::unique_ptr<HistTable>     m_hist;
    std::unique_ptr<ContHistTable> m_cont_hist;

    HistoryManager() {
        m_hist      = std::make_unique<HistTable>();
        m_cont_hist = std::make_unique<ContHistTable>();
    }

    void update_hist(const Position& pos, const MoveList& quiets, Move best_move, int depth) const
    {
        for (const auto [m, s] : quiets) {
            const auto move  = m;
            const auto moved = pos.piece_at(move.from_sq());

            int& entry = m_hist->at(moved)[move.to_sq()];
            if (m == best_move) {
                entry += std::min(8000, depth * depth);
            } else {
                entry -= entry / 10;
            }
        }
    }

    void update_cont_hist(std::span<const Position> positions,
                          const MoveList& quiets, const Move best_move,
                          const int depth, const int max_back = 2) const
    {
        const int end = positions.size() - 1;
        for (int back = 0; back < max_back; ++back) {

            const auto& prev_pos = positions[end - back];
            const auto prev_moved = prev_pos.moved();
            const auto prev_move  = prev_pos.move();

            for (const auto [m, s] : quiets) {
                const auto move  = m;
                const auto moved = positions.back().piece_at(move.from_sq());
                if (prev_move == Move::null()) continue;

                int& entry = m_cont_hist->at(prev_moved)[prev_move.to_sq()][moved][move.to_sq()];

                const int scale = std::max(1, depth * depth / ((2 + back ) / 2));

                if (m == best_move) {
                    entry += std::min(8000, scale);
                } else {
                    entry -= entry / 10;
                }
            }
        }
    }

    [[nodiscard]] int get_hist_bonus(const Position& pos, const Move move) const {
        const auto moved = pos.piece_at(move.from_sq());
        return m_hist->at(moved)[move.to_sq()];
    }


    [[nodiscard]] int get_cont_hist_bonus(const std::span<const Position> positions,
                                          const Move move,
                                          const int max_back = 2) const
    {
        int bonus = 0;
        const int end = positions.size() - 1;

        for (int back = 0; back < max_back; ++back) {

            const auto& prev_pos = positions[end - back];
            const auto prev_moved = prev_pos.moved();
            const auto prev_move  = prev_pos.move();
            if (prev_move == Move::null()) continue;

            const auto moved = positions.back().piece_at(move.from_sq());

            const int entry = m_cont_hist->at(prev_moved)[prev_move.to_sq()][moved][move.to_sq()];
            bonus += entry / ((back + 2) / 2);
        }

        return bonus;
    }
};

#endif // HISTORY_H
