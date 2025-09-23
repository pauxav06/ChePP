#ifndef MOVE_ORDERING_H
#define MOVE_ORDERING_H

#include "history.h"
#include "search_stack.h"
#include "types.h"



struct MoveSelector {

    enum class Stage {
        Root,
        Search,
        QSearch,
        ProbCut
    };

    struct Params {
        int tt_bonus       = 20'000'000;
        int killer1_bonus  = 9'000'000;
        int killer2_bonus  = 8'000'000;
        int good_capture   = 10'000'000;
        int probcut_capture= 10'000'000;
        int root_bonus     = 10'000'000;
    };

    template <std::ranges::range Range>
    MoveSelector(Range&& moves,
                 const Stage stage,
                 const SearchStack::Node& ss,
                 const Params& params,
                 const Move tt_move = Move::none())
    : m_stage(stage),
    m_ss(ss),
    m_params(params),
    m_tt_move(tt_move),
    m_list(moves, [this](const Move m){ return score_move(m); })
    {
        m_remaining = m_list.size();
    }

    [[nodiscard]] bool has_next() const { return m_remaining > 0; }
    [[nodiscard]] bool empty() const { return m_remaining == 0; }
    [[nodiscard]] size_t remaining() const { return m_remaining; }
    [[nodiscard]] size_t size() const { return m_list.size(); }

    std::pair<Move, MoveScoreT> next() {
        assert(has_next());

        size_t best_idx = 0;
        MoveScoreT best_score = m_list[0].score;
        for (size_t i = 1; i < m_remaining; ++i) {
            if (m_list[i].score > best_score) {
                best_score = m_list[i].score;
                best_idx = i;
            }
        }

        --m_remaining;
        std::swap(m_list[best_idx], m_list[m_remaining]);
        return {m_list[m_remaining].move, m_list[m_remaining].score};
    }

    struct iterator {
        using value_type = std::pair<Move, MoveScoreT>;
        using reference  = value_type;
        using pointer    = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        explicit iterator(MoveSelector* stream) : m_stream(stream) {}

        value_type operator*() const { return m_stream->next(); }

        iterator& operator++() { return *this; }
        void operator++(int) { ++(*this); }

        bool operator!=(const iterator&) const { return m_stream && m_stream->has_next(); }

    private:
        MoveSelector* m_stream;
    };

    iterator begin() { return iterator(this); }
    iterator end()   { return iterator(nullptr); }

    [[nodiscard]] const Params& params() const { return m_params; }

private:
    MoveScoreT score_move(Move m) const {
        if (m == m_tt_move) return m_params.tt_bonus;

        const Piece attacker = m_ss.position->piece_at(m.from_sq());
        const Piece victim = m_ss.position->is_capture(m) ? m_ss.position->captured_by_move(m) : NO_PIECE;

        MoveScoreT score = 0;

        switch (m_stage) {
            case Stage::Root: {
                const auto max_nodes = std::ranges::max_element(
                    *m_ss.refutation_history,
                    {},
                    &std::pair<const Move, size_t>::second
                );
                if (max_nodes != m_ss.refutation_history->end()) {
                    score = m_params.root_bonus *
                            m_ss.refutation_history->at(m) / max_nodes->second;
                }
                break;
            }

            case Stage::Search: {
                if (m == m_ss.killer1) { score += m_params.killer1_bonus; break; }
                if (m == m_ss.killer2) { score += m_params.killer2_bonus; break; }
                if (m_ss.position->is_quiet(m)) {
                    score += m_ss.history->at(m_ss.position(), m);
                    if (m_ss.continuation_history) score += m_ss.continuation_history->at(m_ss.position(), m);
                    if (m_ss.prev && m_ss.prev->continuation_history) score += m_ss.prev->continuation_history->at(m_ss.position(), m);
                    break;
                }
            }
            [[fallthrough]];
            case Stage::ProbCut: {
            }
            [[fallthrough]];
            case Stage::QSearch: {
                if (victim) {
                    score += mvv_lva(attacker.type(), victim.type()) + HistoryBonus::max() +
                        m_params.good_capture * (m_ss.position->see(m) > -107) +
                        m_ss.capture_history->at(m_ss.position(), m);
                }
            }
        }
        return score;
    }

    static int mvv_lva(const PieceType attacker, const PieceType victim) {
        return victim.index() * 10 + (KING.index() - attacker.index());
    }

    Stage m_stage;
    const SearchStack::Node& m_ss;
    const Params& m_params;
    Move m_tt_move;
    ScoredMoveList m_list;
    size_t m_remaining{0};
};

#endif // MOVE_ORDERING_H
