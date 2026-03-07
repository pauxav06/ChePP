#ifndef CHEPP_MDSPAN_H
#define CHEPP_MDSPAN_H

#if defined(USE_STD_MDSPAN)
#include <mdspan>
#define MD_ACCESS(c, ...) c[__VA_ARGS__]
namespace md = std;
#else
#include <experimental/mdspan>
#if MDSPAN_USE_BRACKET_OPERATOR
#define MD_ACCESS(c, ...) c[__VA_ARGS__]
#else
#define MD_ACCESS(c, ...) c(__VA_ARGS__) // workaround for apple clang
#endif
namespace md = std::experimental;
#endif

#endif // CHEPP_MDSPAN_H
