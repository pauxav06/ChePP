#ifndef MOVE_ORDERING_H
#define MOVE_ORDERING_H

#include "history.h"
#include "search_stack.h"
#include "types.h"


struct ScoredMove {
    Move move;
    int64_t score;
};

struct ScoredMoveList : ArrayStack<ScoredMove, MoveList::capacity()> {
    using Base = ArrayStack;
    using Base::push_back;
    using Base::operator[];
};

template <typename F>
concept MoveScorer = requires(F f, Move m) {
    { f(m) } -> std::convertible_to<BonusT>;
};



struct ScoredMoveStream {
    explicit ScoredMoveStream(const ScoredMoveList& moves) : m_list(moves) {
        m_remaining = m_list.size();
    }
    ScoredMoveStream() = default;


    bool has_next() const { return m_remaining > 0; }
    bool empty() const { return m_remaining == 0; }

    std::pair<Move, int64_t> next() {
        assert(has_next());

        size_t best_idx = 0;
        int64_t best_score = m_list[0].score;
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

    size_t remaining() const { return m_remaining; }
    size_t size() const { return m_list.size(); }

    struct iterator {
        using value_type = std::pair<Move, int64_t>;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        iterator(ScoredMoveStream* stream) : m_stream(stream) {}

        value_type operator*() const {
            return m_stream->next();
        }

        iterator& operator++() {
            return *this;
        }

        bool operator!=(const iterator& other) const {
            return m_stream->has_next();
        }

    private:
        ScoredMoveStream* m_stream;
    };

    iterator begin() { return iterator(this); }
    iterator end()   { return iterator(nullptr); }

private:
    ScoredMoveList m_list;
    size_t m_remaining{0};
};

template <std::ranges::range R, MoveScorer ScoreFn>
ScoredMoveStream make_scored_stream(R&& moves, ScoreFn&& scorer) {
    ScoredMoveList out;
    for (auto&& m : moves) {
        out.push_back({m, static_cast<int64_t>(scorer(m))});
    }
    return ScoredMoveStream{out};
}

enum class ScorerType { Root, Search, QSearch };

template <ScorerType T>
struct Scorer;


template <>
struct Scorer<ScorerType::Search> {
    struct Params {
        int m_tt_bonus = 1500;
        int m_killer_1_bonus = 1400;
        int m_killer_2_bonus = 1300;
        int promotion_bonus = 1;
        int see_factor = 1;
        int capture_hist_factor = 1;
        int hist_factor = 1;
        int cont_hist_factor = 1;
        int prev_cont_hist_factor = 1;
    };

    Scorer(const Params& params, const SearchStack::Node& ss, const Move tt_move)
        : m_params(params), m_ss(ss), m_tt_move(tt_move) {}

    auto operator()() const {
        return [this](const Move m) -> BonusT {
            BonusT score = 0;
            if (m == m_tt_move) score += m_params.m_tt_bonus;
            if (m == m_ss.killer1) score += m_params.m_killer_1_bonus;
            if (m == m_ss.killer2) score += m_params.m_killer_2_bonus;

            const auto victim = m.type_of() == EN_PASSANT ? PAWN : m_ss.position->piece_at(m.to_sq()).type();

            if (m.type_of() == PROMOTION)
                score += (m.promotion_type().piece_value()) * m_params.promotion_bonus;

            if (victim)
                score += m_ss.position->see(m) * m_params.see_factor + m_ss.capture_history.get_bonus(m);
            else if (m.type_of() != PROMOTION) {
                score += m_ss.history.get_bonus(m);
                if (m_ss.ply > 1 && m_ss.position->move() != Move::null()) score += m_ss.continuation_history.get_bonus(*m_ss.position, m);
                if (m_ss.ply > 2 && m_ss.prev->position->move() != Move::null()) score += m_ss.prev->continuation_history.get_bonus(*m_ss.position, m);
            }

            return score;
        };
    }

private:
    const Params& m_params;
    const SearchStack::Node& m_ss;
    Move m_tt_move;
};


template <>
struct Scorer<ScorerType::Root> {
    struct Params {
        int m_tt_bonus = 1500;
        int max_root_bonus = 1300;
    };

    Scorer(const Params& params, const SearchStack::Node& ss, Move tt_move)
        : m_params(params), m_ss(ss), m_tt_move(tt_move) {}

    auto operator()() const {
        return [this](Move m) -> BonusT {
            BonusT score = 0;
            const auto max_nodes = std::ranges::max_element(
                *m_ss.refutation_nodes,
                {},
                &std::pair<const Move, size_t>::second
            );
            assert(max_nodes != m_ss.refutation_nodes->end());

            if (m == m_tt_move) score = m_params.m_tt_bonus;
            else score = m_params.max_root_bonus * m_ss.refutation_nodes->at(m) / max_nodes->second;

            return score;
        };
    }

private:
    const Params& m_params;
    const SearchStack::Node& m_ss;
    Move m_tt_move;
};

template <>
struct Scorer<ScorerType::QSearch> {
    struct Params {
        int m_tt_bonus = 1500;
        int m_killer_1_bonus = 1400;
        int m_killer_2_bonus = 1300;
        int promotion_bonus = 1;
        int see_factor = 1;
        int capture_hist_factor = 2;
    };

    Scorer(const Params& params, const SearchStack::Node& ss, Move tt_move)
        : m_params(params), m_ss(ss), m_tt_move(tt_move) {}

    auto operator()() const {
        return [this](Move m) -> BonusT {
            BonusT score = 0;
            if (m == m_tt_move) score += m_params.m_tt_bonus;
            if (m == m_ss.killer1) score += m_params.m_killer_1_bonus;
            if (m == m_ss.killer2) score += m_params.m_killer_2_bonus;

            const auto victim = m.type_of() == EN_PASSANT ? PAWN : m_ss.position->piece_at(m.to_sq()).type();

            if (m.type_of() == PROMOTION)
                score += (m.promotion_type().piece_value()) * m_params.promotion_bonus;

            if (victim)
                score += m_ss.position->see(m) * m_params.see_factor + m_ss.capture_history.get_bonus(m);

            return score;
        };
    }

private:
    const Params& m_params;
    const SearchStack::Node& m_ss;
    Move m_tt_move;
};

#endif // MOVE_ORDERING_H
