#ifndef CHEPP_ACCUMULATOR_H_
#define CHEPP_ACCUMULATOR_H_

#include "layers.h"

namespace chepp::nnue::layers::accumulator {
    using Scalar = Kernel<0>;
    using Simd   = Kernel<2>;
} // namespace chepp::nnue::layers::accumulator

#endif // CHEPP_ACCUMULATOR_H
