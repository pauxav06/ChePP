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
    for (auto& [move, score] : list) {

        if (move == prev_best) {
            score = 10000;
        } else if (move == ssNode.killer1) {
            score = 9000;
        } else if (move == ssNode.killer2) {
            score = 8900;
        } else if (move.type_of() == PROMOTION) {
            score = move.promotion_type().piece_value() * 8;
        } else {

            if (auto victim = ss.pos->piece_at(move.to_sq()); victim != NO_PIECE || move.type_of() == EN_PASSANT) {
                score = ss.pos->see(move) * 10;
            } else {
                score += history.get_cont_hist_bonus(ss, move) + history.hist_score(ss, move);
            }
        }
    }
}


#endif // MOVE_ORDERING_H
