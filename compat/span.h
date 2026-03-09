//
// Created by paul on 3/9/26.
//

#ifndef CHEPP_SPAN_H
#define CHEPP_SPAN_H

#if defined(USE_STD_SPAN)
#include <span>
namespace tcb = std;
#else
#include <tcb/span.hpp>
#endif

#endif // CHEPP_SPAN_H
