#ifndef MOVE_ORDERING_H
#define MOVE_ORDERING_H

#include "history.h"
#include "search_stack.h"
#include "types.h"

inline void score_moves(const SearchStack::Node& ss,
                        MoveList& list,
                        const Move prev_best,
                        HistoryManager& history,
                        const SearchStack::Node& ssNode
                        )
{
    for (auto& [move, score] : list)
    {
        score  = 0;
        if (move == prev_best) {
            score += 500'000'000;
        } if (move == ssNode.killer1) {
            score += 80'000'000;
        } if (move == ssNode.killer2) {
            score += 79'000'000;
        }
        auto victim = move.type_of() == EN_PASSANT ? PAWN : ss.pos->piece_at(move.to_sq()).type();
        if (move.type_of() == PROMOTION)
            score += (move.promotion_type().piece_value()) * 100'000;
        if (victim)
            score += ss.pos->see(move) * 100'000 + history.get_capture_hist_score(ss, move);
        if (!victim && move.type_of() != PROMOTION)
        {
            score += history.get_cont_hist_bonus(ss, move);
            score +=  history.get_hist_score(ss, move);
        }
    }
}


#endif // MOVE_ORDERING_H
