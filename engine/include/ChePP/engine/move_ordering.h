#ifndef MOVE_ORDERING_H
#define MOVE_ORDERING_H

#include "history.h"
#include "search_stack.h"
#include "types.h"

template <typename F>
concept MoveScorer = requires(F f, Move m) {
    { f(m) } -> std::convertible_to<MoveScoreT>;
};

struct ScoredMoveStream {

    struct ScoredMove {
        Move move;
        MoveScoreT score;
    };

    struct ScoredMoveList : ArrayStack<ScoredMove, MoveList::capacity()> {
        using Base = ArrayStack;
        using Base::push_back;
        using Base::operator[];

        template <std::ranges::range R, MoveScorer ScoreFn>
        ScoredMoveList(R&& moves, ScoreFn&& scorer)
        {
            for (auto&& m : moves) {
                push_back({m, scorer(m)});
            }
        }
    };

    explicit ScoredMoveStream(const ScoredMoveList& moves) : m_list(moves), m_remaining(moves.size()) {}
    template <std::ranges::range R, MoveScorer ScoreFn>
    ScoredMoveStream(R&& moves, ScoreFn&& scorer) : ScoredMoveStream(ScoredMoveList(moves, scorer)) {}

    [[nodiscard]] bool has_next() const { return m_remaining > 0; }
    [[nodiscard]] bool empty() const { return m_remaining == 0; }

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

    [[nodiscard]] size_t remaining() const { return m_remaining; }
    [[nodiscard]] size_t size() const { return m_list.size(); }

    struct iterator {
        using value_type = std::pair<Move, int64_t>;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        explicit iterator(ScoredMoveStream* stream) : m_stream(stream) {}

        value_type operator*() const {
            return m_stream->next();
        }

        iterator& operator++() {
            return *this;
        }

        bool operator!=(const iterator& _) const {
            return m_stream->has_next();
        }

    private:
        ScoredMoveStream* m_stream;
    };

    iterator begin() { return iterator(this); }
    static iterator end()   { return iterator(nullptr); }

private:
    ScoredMoveList m_list;
    size_t m_remaining{0};
};



inline int mvv_lva(const PieceType attacker, const PieceType victim)
{
    return ((victim.index()) * 100) + (KING.index() - attacker.index());
}

enum class ScorerType { Root, Search, QSearch };

struct RootParams {
    int m_tt_bonus = 20000000;
    int max_root_bonus = 10000000;
};

struct SearchParams {
    int m_tt_bonus = 20000000;
    int m_killer_1_bonus = 9000000;
    int m_killer_2_bonus = 8000000;
    int good_capture_bonus = 10000000;
};

struct QSearchParams {
    int m_tt_bonus = 20000000;
    int good_capture_bonus = 10000000;
    int m_killer_1_bonus = 9000000;
    int m_killer_2_bonus = 8000000;
};

template <ScorerType T>
struct ParamsFor;

template <>
struct ParamsFor<ScorerType::Root> { using type = RootParams; };

template <>
struct ParamsFor<ScorerType::Search> { using type = SearchParams; };

template <>
struct ParamsFor<ScorerType::QSearch> { using type = QSearchParams; };

template <ScorerType T>
struct Scorer {
    using Params = ParamsFor<T>::type;

    Scorer(const Params& params, const SearchStack::Node& ss, Move tt_move)
        : m_params(params), m_ss(ss), m_tt_move(tt_move) {}

    auto operator()() const {
        return [this](Move m) -> MoveScoreT {
            MoveScoreT score = 0;

            if constexpr (T == ScorerType::Root) {
                const auto max_nodes = std::ranges::max_element(
                    *m_ss.refutation_nodes,
                    {},
                    &std::pair<const Move, size_t>::second
                );
                assert(max_nodes != m_ss.refutation_nodes->end());

                if (m == m_tt_move)
                    score = m_params.m_tt_bonus;
                else
                    score = m_params.max_root_bonus * m_ss.refutation_nodes->at(m) / max_nodes->second;
            } else {
                const auto attacker = m_ss.position->piece_at(m.from_sq()).type();
                const auto victim = m.type_of() == EN_PASSANT ? PAWN : m_ss.position->piece_at(m.to_sq()).type();

                if (m == m_tt_move) score += m_params.m_tt_bonus;
                else if constexpr (T == ScorerType::Search) {
                    if (m == m_ss.killer1) score += m_params.m_killer_1_bonus;
                    else if (m == m_ss.killer2) score += m_params.m_killer_2_bonus;

                    else if (victim)
                        score += mvv_lva(attacker, victim) + m_params.good_capture_bonus * (m_ss.position->see(m) > -107);
                    else {
                        score += m_ss.history.get_bonus(m);
                        if (m_ss.ply > 1) score += m_ss.continuation_history.get_bonus(m_ss.position(), m);
                        if (m_ss.ply > 2) score += m_ss.prev->continuation_history.get_bonus(m_ss.position(), m);
                    }
                } else if constexpr (T == ScorerType::QSearch) {
                    if (victim)
                        score += mvv_lva(attacker, victim) + m_params.good_capture_bonus * (m_ss.position->see(m) > -107);
                }
            }

            return score;
        };
    }

    const Params& m_params;
    const SearchStack::Node& m_ss;
    Move m_tt_move;
};


#endif // MOVE_ORDERING_H
