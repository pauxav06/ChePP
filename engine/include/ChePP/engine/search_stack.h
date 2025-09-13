//
// Created by paul on 9/13/25.
//

#ifndef CHEPP_SEARCH_STACK_H
#define CHEPP_SEARCH_STACK_H


#include <cassert>
#include <cstddef>
#include <memory>

#include "ChePP/engine/position.h"

class SearchStack {
public:
    class Node {
    public:
        // general infos
        Move move{Move::null()};
        int ply{0};
        int eval{0};
        Move excluded{Move::none()};

        // heuristics
        Move killer1{Move::none()};
        Move killer2{Move::none()};



        Node* next() {
            if (const auto self = this; self + 1 < owner_end_) {
                return self + 1;
            }
            return nullptr;
        }

        Node* prev() {
            if (const auto self = this; self > owner_begin_) {
                return self - 1;
            }
            return nullptr;
        }


    private:
        friend class SearchStack;
        inline static Node* owner_begin_{nullptr};
        inline static Node* owner_end_{nullptr};
    };

    explicit SearchStack(const std::size_t depth)
        : capacity_(depth),
          nodes_(std::make_unique<Node[]>(depth))
    {
        Node::owner_begin_ = nodes_.get();
        Node::owner_end_   = nodes_.get() + depth;
    }

    Node& operator[](std::size_t i) {
        assert(i < capacity_);
        return nodes_[i];
    }

    const Node& operator[](const std::size_t i) const {
        assert(i < capacity_);
        return nodes_[i];
    }

    [[nodiscard]] std::size_t capacity() const { return capacity_; }

private:
    std::size_t capacity_;
    std::unique_ptr<Node[]> nodes_;
};



#endif // CHEPP_SEARCH_STACK_H
