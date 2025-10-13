#include "tune.h"
#include "utils.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "affine_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"
#include <hwy/base.h>

#include "affine-inl.h"
#include "relu-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::tests::nnue::layers::affine
{
    namespace HWY_NAMESPACE
    {
        namespace
        {

            namespace hn = hwy::HWY_NAMESPACE;
            namespace an = chepp::nnue::layers::affine::HWY_NAMESPACE;

            using namespace hn;
            using namespace meta;
            using namespace chepp::nnue::layers::affine;

            void affine()
            {
                constexpr size_t in         = 2048;
                constexpr size_t out        = 16;
                constexpr size_t Iterations = 10000;

                hwy::AlignedVector<int8_t>  input(in);
                hwy::AlignedVector<int8_t>  weights(in * out);
                hwy::AlignedVector<int32_t> biases(out);
                hwy::AlignedVector<int32_t> ref_output(out);
                hwy::AlignedVector<int32_t> simd_output(out);

                utils::fill_random(weights);
                utils::fill_random(biases);
                utils::fill_random(input);

                constexpr Types types = {
                    .in  = ScalarType::Int8,
                    .out = ScalarType::Int32,
                };

                constexpr Dims dims = {
                    .in  = in,
                    .out = out,
                };

                constexpr Params<Kernels::Scalar> ref_params{types, dims, {}};
                using reference_layer_t = an::Layer<Kernels::Scalar, ref_params>;
                reference_layer_t reference_layer;

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

                reference_layer.load_weights(weights.data(), biases.data());

                auto simd_layers = [&]<size_t... I>(std::index_sequence<I...>)
                {
                    return std::make_tuple((
                        [&]
                        {
                            using layer_t = an::Layer<Kernels::SIMD, simd_params[I]>;
                            return layer_t{};
                        }())...);
                }(std::make_index_sequence<simd_params.size()>{});

                std::apply(
                    [&](auto&&... layer)
                    {
                        ((
                             [&]()
                             {
                                 layer.forward(input.data(), simd_output.data());
                                 HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), simd_output.size());
                             }()),
                         ...);
                    },
                    simd_layers);
            }
        } // namespace
    } // namespace HWY_NAMESPACE
} // namespace chepp::tests::nnue::layers::affine

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::tests::nnue::layers::affine
{
    namespace
    {
        HWY_BEFORE_TEST(AffineTest);
        HWY_EXPORT_AND_TEST_P(AffineTest, affine);
    } // namespace
} // namespace chepp::tests::nnue::layers::affine
HWY_TEST_MAIN();
#endif // HWY_ONCE