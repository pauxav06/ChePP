#ifndef CHEPP_AFFINE_H
#define CHEPP_AFFINE_H

#include "layers.h"
#include "utils.h"

namespace chepp::nnue::layers::affine
{
    using namespace meta;

    enum class Kernels {Scalar, SIMD};

    struct Types
    {
        ScalarType in;
        ScalarType out;
    };

    struct Dims
    {
        size_t in;
        size_t out;
    };

    template <Kernels K>
    struct Opt;

    template <>
    struct Opt<Kernels::Scalar> {
    };

    template <>
    struct Opt<Kernels::SIMD> {
        enum Operation : uint8_t {SumOfMulQuadAcc, SumOfMulPairAdd};
        size_t unroll;
        Operation operation;
    };

    template <Kernels K>
    struct Params
    {
        static constexpr Kernels kernel = K;
        Types types;
        Dims dims;
        Opt<kernel> opt;
    };
}

#endif // CHEPP_AFFINE_H
