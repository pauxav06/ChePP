//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_BINPACK_CONVERTER_H
#define CHEPP_BINPACK_CONVERTER_H
#include <utility>

template <typename Callable>
struct BinpackConverter {
    Callable func;
    auto operator()(auto&& entry) const {
        return func(std::forward<decltype(entry)>(entry));
    }
};

template<typename Callable>
BinpackConverter(Callable) -> BinpackConverter<Callable>;


#endif // CHEPP_BINPACK_CONVERTER_H
