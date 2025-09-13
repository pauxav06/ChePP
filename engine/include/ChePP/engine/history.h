//
// Created by paul on 9/11/25.
//

#ifndef HISTORY_H
#define HISTORY_H

#include "position.h"
#include "search_stack.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

template <typename T>
using HistTableT = EnumArray<Piece, EnumArray<Square, T>>;

using HistTable = HistTableT<int>;
using ContHistTable = HistTableT<HistTableT<int>>;

struct HistoryManager {


    std::unique_ptr<HistTable>     m_hist{};
    std::unique_ptr<ContHistTable> m_cont_hist{};

    HistoryManager() {
        m_hist      = std::make_unique<HistTable>();
        m_cont_hist = std::make_unique<ContHistTable>();
    }

    HistTable& hist() { return *m_hist; }
    ContHistTable& cont_hist() { return *m_cont_hist; }


    HistTable& cont_hist(const SearchStack::Node& ss) {
        return cont_hist()[ss.pos->moved()][ss.pos->move().to_sq()];
    }

    int& hist_score(const SearchStack::Node& ss, const Move move)
    {
        return hist()[ss.pos->piece_at(move.from_sq())][move.to_sq()];
    }

    int& cont_hist_score(const SearchStack::Node& ss, const Move move)
    {
        return cont_hist(ss)[ss.pos->piece_at(move.from_sq())][move.to_sq()];
    }


    void update_hist(const SearchStack::Node& ss, const Move move, const std::function<int(int)>& func)
    {
        hist_score(ss, move) = std::clamp(func(hist_score(ss, move)), 0, 6000);
    }

    void update_cont_hist(const SearchStack::Node& ss, const Move move, const std::function<int(int)>& func)
    {
        cont_hist_score(ss, move) = std::clamp(func(cont_hist_score(ss, move)), 0, 6000);
    }

    void update_hist(const SearchStack::Node& ss, const MoveList& quiets, const Move best_move, const int depth)
    {
        for (const auto [m, s] : quiets) {

            if (m == best_move) {
                update_hist(ss, m, [&] (const int score) {return score + depth * depth; });
            } else {
                update_hist(ss, m, [&] (const int score) {return score - score / 5; });
            }
        }
    }

    void update_cont_hist(const SearchStack::Node& ss_init,
                          const MoveList& quiets, const Move best_move,
                          const int depth, const int max_back = 2)
    {
        const SearchStack::Node* ss = &ss_init;

        for (int back = 0; back < max_back && ss; ++back, ss = ss_init.prev()) {

            for (const auto [m, s] : quiets) {
                const auto move  = m;
                if (ss->pos->move() == Move::null()) continue;

                const int scale = std::max(1, depth * depth / ((2 + back ) / 2));

                if (m == best_move) {
                    update_cont_hist(*ss, move, [&] (const int score) {return score + depth * depth / scale; });
                } else {
                    update_cont_hist(*ss, move, [&] (const int score) {return score - score / (5 * scale); });
                }
            }
        }
    }



    [[nodiscard]] int get_cont_hist_bonus(const SearchStack::Node& ss_init,
                                          const Move move,
                                          const int max_back = 2)
    {
        int bonus = 0;

        const SearchStack::Node* ss = &ss_init;

        for (int back = 0; back < max_back && ss; ++back, ss = ss_init.prev()) {
            bonus += cont_hist_score(*ss, move) / ((back + 2) / 2);
        }

        return bonus;
    }
};

#endif // HISTORY_H
