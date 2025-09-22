//
// Created by paul on 9/13/25.
//

#ifndef CHEPP_SEARCH_STACK_H
#define CHEPP_SEARCH_STACK_H

#include "nnue.h"
#include "history.h"

#include <cassert>
#include <cstddef>
#include <memory>

#include "ChePP/engine/position.h"



class SearchStack {
public:
    struct Node
    {
        Node* prev{};
        int ply{};
        Positions::Handle position{};
        Accumulators::Handle accumulator{};
        bool is_repetition{false};

        int static_eval{0};
        Move excluded{Move::none()};
        int double_extensions{0};
        int extensions{0};
        Move best_move{Move::none()};

        Move killer1{Move::none()};
        Move killer2{Move::none()};

        ContinuationHistory continuation_history{};
        History history{};
        History capture_history{};
        RefutationMapT* refutation_nodes{};
    };

    explicit SearchStack(const Positions& positions)
    : m_positions(positions),
    m_accumulators(positions.last()),
    m_nodes(std::make_unique<Node[]>(MAX_PLY)),
    m_history(std::make_unique<HistoryTable>()),
    m_capture_history(std::make_unique<HistoryTable>()),
    m_continuation_history(std::make_unique<ContinuationHistoryTable>())
    {
        update_last_node();
    }

    [[nodiscard]] int ply() const { return m_positions.ply(); }

    Node& operator[](const int i) {
        assert(i >= 0 && i <= ply());
        return m_nodes[i];
    }

    const Node& operator[] (const int i) const {
        assert(i >= 0 && i <= ply());
        return m_nodes[i];
    }

    void do_move(const Move move, const bool update_nnue)
    {
        const int init_ply = ply();
        m_positions.do_move(move);
        if (update_nnue && move != Move::none() && move != Move::null())
        {
            m_accumulators.do_move(m_positions[init_ply], m_positions[ply()]);
        }
        update_last_node();
    }

    void undo_move(const bool update_nnue)
    {
        assert(ply() > 0);
        const Move move = m_positions.last().move();
        reset_last_node();
        if (update_nnue && move != Move::none() && move != Move::null())
        {
            m_accumulators.undo_move();
        }
        m_positions.undo_move();
    }

private:
    void update_last_node()
    {
        Node& node = m_nodes[ply()];
        node.prev = ply() == 0 ? nullptr : &m_nodes[ply() - 1];
        node.ply = ply();
        node.accumulator = m_accumulators.handle_to_last();
        node.position = m_positions.handle_to_last();
        node.continuation_history = ContinuationHistory(m_continuation_history.get(), node.position);
        node.history = History(m_history.get(), node.position);
        node.capture_history = History(m_capture_history.get(), node.position);
        node.refutation_nodes = &m_refutation_nodes;
        node.is_repetition = m_positions.is_repetition();
        node.extensions = node.prev ? m_nodes[ply() - 1].extensions : 0;
        node.double_extensions = node.prev ? m_nodes[ply() - 1].double_extensions : 0;
    }

    void reset_last_node()
    {
        Node& node = m_nodes[ply()];
        node.prev = nullptr;
        node.ply = 0;
        node.position = {};
        node.accumulator = {};
        node.history = {};
        node.continuation_history = {};
        node.capture_history = {};
        node.refutation_nodes = nullptr;
        node.is_repetition = false;
        node.double_extensions = 0;
        node.extensions = 0;
    }

    Positions m_positions;
    Accumulators m_accumulators;
    std::unique_ptr<Node[]> m_nodes;

    std::unique_ptr<HistoryTable> m_history{};
    std::unique_ptr<HistoryTable> m_capture_history{};
    std::unique_ptr<ContinuationHistoryTable> m_continuation_history{};
    RefutationMapT m_refutation_nodes{MAX_MOVES};
};



#endif // CHEPP_SEARCH_STACK_H
