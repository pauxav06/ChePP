#ifndef CHEPP_RELU_H_
#define CHEPP_RELU_H_

#include "layers.h"
#include "types.h"
#include "utils.h"

namespace chepp::nnue::layers::relu {
    using Scalar = Kernel<0>;
    using Simd   = Kernel<1>;

} // namespace chepp::nnue::layers::relu

#endif // CHEPP_RELU_H
