#ifndef CHEPP_AFFINE_H
#define CHEPP_AFFINE_H

#include "layers.h"
#include "types.h"
#include "utils.h"
#include <hwy/aligned_allocator.h>

namespace chepp::nnue::layers::affine {
    using Scalar = Kernel<0>;
    // using SimdRowMaj = Kernel<1>; should implement later
    using SimdColMaj = Kernel<2>;

    using SumOfMulQuadAcc = Operation<0>;
    using SumOfMulPairAcc = Operation<1>;

    struct AffineParams {
        Dims dims;
    };

    template <typename InT, typename OutT>
    struct StateParams {
        int             bucket;
        std::span<InT>  weights;
        std::span<OutT> biases;
    };

    template <typename InT, typename OutT>
    struct AffineState {
        using input_type  = InT;
        using output_type = OutT;
        hwy::AlignedVector<input_type>  weights;
        hwy::AlignedVector<output_type> biases;

        AffineState(size_t w_size, size_t b_size) : weights(w_size), biases(b_size) {}
    };

} // namespace chepp::nnue::layers::affine

#endif // CHEPP_AFFINE_H
