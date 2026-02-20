//
// Created by paul on 9/13/25.
//

#ifndef CHEPP_SEARCH_STACK_H
#define CHEPP_SEARCH_STACK_H

#include "history.h"
#include "nnue.h"
#include "position.h"

#include <cassert>
#include <cstddef>
#include <memory>

class SearchStack {
  public:
    // null pointers indicate the desired information does not exist and should be checked
    struct Node {
        Node* prev{};
        int   ply{};

        Positions::Handle           position{};
        chepp::nnue::Arch::Network* network{};

        bool is_repetition{false};
        int  static_eval{0};

        Move excluded{Move::none()};
        int  single_extensions{0};
        int  double_extensions{0};

        Move best_move{Move::none()};

        Move               killer1{Move::none()};
        Move               killer2{Move::none()};
        History*           continuation_history{};
        History*           history{};
        CaptureHistory*    capture_history{};
        RefutationHistory* refutation_history{};
    };

    explicit SearchStack(const Positions& positions)
        : m_positions(positions), m_network(chepp::nnue::Arch::make_network()),
          m_nodes(std::make_unique<Node[]>(MAX_PLY)), m_history(std::make_unique<History>()),
          m_capture_history(std::make_unique<CaptureHistory>()),
          m_continuation_history(std::make_unique<ContinuationHistory>()),
          m_null_move_continuation_history(std::make_unique<History>()) {
        m_network.init(m_positions.last());
        update_last_node();
    }

    [[nodiscard]] int
    ply() const {
        return m_positions.ply();
    }

    Node&
    operator[](const int i) {
        assert(i >= 0 && i <= ply());
        return m_nodes[i];
    }

    const Node&
    operator[](const int i) const {
        assert(i >= 0 && i <= ply());
        return m_nodes[i];
    }

    void
    do_move(const Move move, const bool update_nnue) {
        const int init_ply = ply();
        m_positions.do_move(move);
        if (update_nnue && move != Move::none() && move != Move::null()) {
            m_network.update(m_positions[init_ply], m_positions[ply()]);
        }
        update_last_node();
    }

    void
    undo_move(const bool update_nnue) {
        assert(ply() > 0);
        const Move move = m_positions.last().move();
        reset_last_node();
        if (update_nnue && move != Move::none() && move != Move::null()) {
            m_network.undo();
        }
        m_positions.undo_move();
    }

  private:
    void
    update_last_node() {
        Node& node = m_nodes[ply()];
        node.prev  = ply() == 0 ? nullptr : &m_nodes[ply() - 1];
        node.ply   = ply();

        node.network       = &m_network;
        node.position      = m_positions.handle_to_last();
        node.is_repetition = m_positions.is_repetition();

        node.single_extensions = node.prev ? m_nodes[ply() - 1].single_extensions : 0;
        node.double_extensions = node.prev ? m_nodes[ply() - 1].double_extensions : 0;

        node.history              = m_history.get();
        node.capture_history      = m_capture_history.get();
        node.continuation_history = !node.prev ? nullptr
                                    : node.position->move() == Move::null()
                                        ? nullptr
                                        : &m_continuation_history->get_relevant_history(node.position());
        node.refutation_history   = &m_refutation_nodes;
    }

    void
    reset_last_node() {
        Node& node              = m_nodes[ply()];
        node.prev               = nullptr;
        node.ply                = 0;
        node.position           = Positions::Handle{nullptr, 0};
        node.network            = nullptr;
        node.history            = nullptr;
        node.capture_history    = nullptr;
        node.refutation_history = nullptr;
        node.is_repetition      = false;
        node.double_extensions  = 0;
        node.single_extensions  = 0;
    }

    Positions                  m_positions;
    chepp::nnue::Arch::Network m_network;
    std::unique_ptr<Node[]>    m_nodes;

    std::unique_ptr<History>             m_history{};
    std::unique_ptr<CaptureHistory>      m_capture_history{};
    std::unique_ptr<ContinuationHistory> m_continuation_history{};
    std::unique_ptr<History>             m_null_move_continuation_history;
    RefutationHistory                    m_refutation_nodes{MAX_MOVES};
};

#endif // CHEPP_SEARCH_STACK_H
