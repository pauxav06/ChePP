//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_STEAM_SOURCE_H
#define CHEPP_STEAM_SOURCE_H

#include <optional>

template<typename T>
struct StreamSource {
    virtual std::optional<T> next() = 0;
    virtual ~StreamSource() = default;
};

#endif // CHEPP_STEAM_SOURCE_H
