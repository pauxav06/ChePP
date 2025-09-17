//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_STREAM_VIEW_H
#define CHEPP_STREAM_VIEW_H

#include "steam_source.h"
#include <ranges>

template<typename T>
class StreamView : public std::ranges::view_base {
    struct iterator {
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using iterator_concept  = std::input_iterator_tag;
        using iterator_category = std::input_iterator_tag;

        StreamSource<T>* src   = nullptr;
        std::optional<T>  current;

        iterator() = default;
        explicit iterator(StreamSource<T>* s) : src(s), current(src->next()) {}

        T operator*() const { return *current; }
        iterator& operator++() { current = src->next(); return *this; }
        void operator++(int) { ++*this; }
        bool operator==(std::default_sentinel_t) const { return !current.has_value(); }
    };

    StreamSource<T>* src;

public:
    explicit StreamView(StreamSource<T>& s) : src(&s) {}
    auto begin() { return iterator{src}; }
    static auto end() { return std::default_sentinel; }
};

#endif // CHEPP_STREAM_VIEW_H
