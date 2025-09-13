#ifndef MOVE_ORDERING_H
#define MOVE_ORDERING_H

#include "history.h"
#include "types.h"


inline void score_moves(const std::span<const Position> positions,
                        MoveList& list,
                        const Move prev_best,
                        const HistoryManager& history,
                        const SearchStackNode& ssNode

                        )
{
    const Position& pos = positions.back();
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
            auto victim   = pos.piece_at(move.to_sq());

            if (victim != NO_PIECE || move.type_of() == EN_PASSANT) {
                score = pos.see(move) * 10;
            } else {
                auto back = std::min(2, int(positions.size()) - 1);
                score += history.get_cont_hist_bonus(positions, move, back) + history.get_hist_bonus(pos, move);
                //score += pos.see(move);
            }
        }
    }
}


#endif // MOVE_ORDERING_H
