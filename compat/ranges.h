#ifndef CHEPP_RANGES_H
#define CHEPP_RANGES_H

#if defined(USE_STD_RANGES)
#include <ranges>
namespace ranges = std::ranges;
#else
#include <range/v3/all.hpp>
#endif

#endif // CHEPP_RANGES_H
