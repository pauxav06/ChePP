#ifndef CHEPP_NETWORK_H_
#define CHEPP_NETWORK_H_

#include "affine.h"
#include "tune.h"

#include <cstddef>

namespace chepp::nnue::network {
    using namespace meta;

    namespace accum {}

    namespace l1 {
        using namespace chepp::nnue::layers::affine;
        constexpr Types types = {
            .in  = ScalarType::Int8,
            .out = ScalarType::Int32,
        };

        constexpr Dims dims = {
            .in  = 1024,
            .out = 16,
        };

        constexpr Params<Kernels::Scalar> ref_params{types, dims, {}};

        using simd_opt_t                     = Opt<Kernels::SIMD>;
        using param_t                        = Params<Kernels::SIMD>;
        constexpr std::array simd_type_arr   = {types};
        constexpr std::array simd_dim_arr    = {dims};
        constexpr std::array simd_unroll_arr = {1, 2, 4, 8, 16};
        constexpr std::array simd_op_arr     = {simd_opt_t::Operation::SumOfMulQuadAcc,
                                                simd_opt_t::Operation::SumOfMulPairAdd};

        constexpr auto simd_params = chepp::nnue::tune::generate_combinations<param_t>(
            simd_type_arr, simd_dim_arr,
            chepp::nnue::tune::generate_combinations<simd_opt_t>(simd_unroll_arr, simd_op_arr));
    } // namespace l1

} // namespace chepp::nnue::network

#endif // CHEPP_NETWORK_INL_H
