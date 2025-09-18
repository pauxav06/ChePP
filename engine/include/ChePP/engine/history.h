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
using ContHistTable = HistTableT<HistTable>;
using CaptureHistTable = EnumArray<PieceType, HistTable>;

struct HistoryManager {

    std::unique_ptr<HistTable>         m_hist{};
    std::unique_ptr<ContHistTable>     m_cont_hist{};
    std::unique_ptr<HistTable>         m_pawn_hist{};
    std::unique_ptr<CaptureHistTable>  m_capture_hist{};

    HistoryManager() {
        m_hist         = std::make_unique<HistTable>();
        m_cont_hist    = std::make_unique<ContHistTable>();
        m_pawn_hist    = std::make_unique<HistTable>();
        m_capture_hist = std::make_unique<CaptureHistTable>();
    }

    template <typename TableT, typename PieceFunc>
    static int& hist_entry(TableT& table, const Move move, const Position& pos, PieceFunc piece_selector) {
        Piece p = piece_selector(pos, move);
        return table[p][move.to_sq()];
    }

    static int& hist_entry(CaptureHistTable& table, const Move move, const Position& pos) {
        Piece attacker = pos.piece_at(move.from_sq());
        PieceType captured = move.type_of() == EN_PASSANT ? PAWN : pos.piece_at(move.to_sq()).type();
        return table[captured][attacker][move.to_sq()];
    }

    static HistTable& cont_hist_entry(ContHistTable& table, const SearchStack::Node& ss) {
        return table[ss.pos->moved()][ss.pos->move().to_sq()];
    }

    template <typename TableT, typename PieceFunc>
    static void update_entry(TableT& table, const Move move, const Position& pos,
                             PieceFunc piece_selector, const std::function<int(int)>& func) {
        int& ref = hist_entry(table, move, pos, piece_selector);
        ref = func(ref);
        ref = std::clamp(ref, 0, 50'000'000);
    }

    static void update_entry(CaptureHistTable& table, const Move move, const Position& pos,
                             const std::function<int(int)>& func) {
        int& ref = hist_entry(table, move, pos);
        ref = std::clamp(ref, 0, 50'000'000);
    }

    void update_hist(const SearchStack::Node& ss, const Move move, const std::function<int(int)>& func) {
        update_entry(*m_hist, move, *ss.pos, [] (const Position& pos, const Move& move) {
            return pos.piece_at(move.from_sq());
        }, func);
    }


    void update_pawn_hist(const SearchStack::Node& ss, const Move move, const std::function<int(int)>& func) {
        update_entry(*m_pawn_hist, move, *ss.pos, [] (const Position& pos, const Move& move) {
            return pos.piece_at(move.from_sq());
        }, func);
    }

    void update_capture_hist(const SearchStack::Node& ss, const Move move, const std::function<int(int)>& func) {
        update_entry(*m_capture_hist, move, *ss.pos, func);
    }

    void update_cont_hist(const SearchStack::Node& ss, const Move move, const std::function<int(int)>& func) {
        HistTable& t = cont_hist_entry(*m_cont_hist, ss);
        update_entry(t, move, *ss.pos, [] (const Position& pos, const Move& move) {
            return pos.piece_at(move.from_sq());
        }, func);
    }

    void update_hist(const SearchStack::Node& ss, const MoveList& quiets, const Move best_move, int depth) {
        for (const auto& [m, _] : quiets) {
            if (m == best_move)
                update_hist(ss, m, [&](int score) { return score + depth * depth * 500; });
            else
                update_hist(ss, m, [&](int score) { return score - score / 50; });
        }
    }

    void update_pawn_hist(const SearchStack::Node& ss, const MoveList& quiets, const Move best_move, int depth) {
        for (const auto& [m, _] : quiets) {
            if (ss.pos->piece_type_at(m.from_sq()) != PAWN) continue;
            if (m == best_move)
                update_pawn_hist(ss, m, [&](int score) { return score + depth * depth * 200; });
            else
                update_pawn_hist(ss, m, [&](int score) { return score - score / 30; });
        }
    }


    void update_capture_hist(const SearchStack::Node& ss, const MoveList& captures, const Move best_move, int depth) {
        for (const auto& [m, _] : captures) {
                if (m == best_move)
                    update_capture_hist(ss, m, [&](int score) { return score + depth * depth * 1000; });
                else
                    update_capture_hist(ss, m, [&](int score) { return score - score / 5; });
        }
    }



    void update_cont_hist(const SearchStack::Node& ss_init, const MoveList& quiets,
                          const Move best_move, int depth, int max_back = 2) {
        const SearchStack::Node* ss = &ss_init;
        for (int back = 0; back < max_back && ss->prev(); ++back, ss = ss->prev()) {
            if (ss->pos->move() == Move::null() || ss->pos->move() == Move::none()) continue;
            for (const auto& [m, _] : quiets) {
                if (m == best_move)
                    update_cont_hist(*ss, m, [&](int score) { return score + depth * depth * 300; });
                else
                    update_cont_hist(*ss, m, [&](int score) { return score - score / 100; });
            }
        }
    }

    [[nodiscard]] int get_cont_hist_bonus(const SearchStack::Node& ss_init,
                                          const Move move, int max_back = 2) const {
        int bonus = 0;
        const SearchStack::Node* ss = &ss_init;
        for (int back = 0; back < max_back && ss->prev(); ++back, ss = ss->prev()) {
            if (ss->pos->move() == Move::null() || ss->pos->move() == Move::none()) continue;
            bonus += hist_entry(cont_hist_entry(*m_cont_hist, *ss), move, *ss->pos, [] (const Position& pos, const Move& move) {
                return pos.piece_at(move.from_sq());
            });
        }
        return bonus;
    }


    int get_hist_score(const SearchStack::Node& ss, const Move move) const {
        const Piece p = ss.pos->piece_at(move.from_sq());
        return (*m_pawn_hist)[p][move.to_sq()];
    }

    int get_pawn_hist_score(const SearchStack::Node& ss, const Move move) const {
        const Piece p = ss.pos->piece_at(move.from_sq());
        return (*m_pawn_hist)[p][move.to_sq()];
    }

    int get_capture_hist_score(const SearchStack::Node& ss, const Move move) const {
        const Piece attacker = ss.pos->piece_at(move.from_sq());
        const PieceType captured = move.type_of()== EN_PASSANT ? PAWN : ss.pos->piece_at(move.to_sq()).type();
        return (*m_capture_hist)[captured][attacker][move.to_sq()];
    }
};

#endif // HISTORY_H
