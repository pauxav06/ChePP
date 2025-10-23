#include "meta.h"
#include "utils.h"

#include <hwy/base.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "affine_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#include "../../build/clang-debug/_deps/highway-build/googletest-src/googletest/include/gtest/gtest-spi.h"
#include "affine-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::tests::nnue::layers::affine {
    namespace HWY_NAMESPACE {
        namespace {

            namespace hn = hwy::HWY_NAMESPACE;
            namespace an = chepp::nnue::layers::affine::HWY_NAMESPACE;

            using namespace hn;
            using namespace chepp::nnue::meta;
            using namespace chepp::nnue::layers;
            using namespace chepp::nnue::layers::affine;

            void inference() {
                using namespace hwy;

                static constexpr size_t input_size  = dims::in;
                static constexpr size_t output_size = dims::out;

                AlignedVector<int8_t>  weights(input_size * output_size);
                AlignedVector<int32_t> biases(output_size);
                AlignedVector<int8_t>  input(input_size);
                AlignedVector<int32_t> ref_output(output_size);

                utils::fill_random(weights);
                utils::fill_random(biases, 0, 0);
                utils::fill_random(input, 0);

                using types = Types<int8_t, int32_t>;

                using scalar_params     = ParamComb_t<Scalar, types, dims>;
                using reference_layer_t = an::Layer<std::tuple_element_t<0, scalar_params>>;
                reference_layer_t reference_layer{};

                using UnrollOptions = std::tuple<Unroll<1>, Unroll<8>>;
                using DotOps        = std::tuple<SumOfMulQuadAcc, SumOfMulPairAcc>;
                using simd_ops      = ParamComb_t<SimdColMaj, types, dims, UnrollOptions, DotOps>;

                reference_layer.load_weights(weights, biases);
                reference_layer.forward(input, ref_output);

                constexpr int rep = 10000;

                for_each_type<simd_ops>([&]<typename Opt>() {
                    auto thread_func = [&]() {
                        AlignedVector<int32_t> simd_output(output_size);
                        for (int i = 0; i < rep; ++i) {
                            an::Layer<Opt> layer;
                            layer.load_weights(weights, biases);
                            layer.forward(input, simd_output);
                            HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), simd_output.size());
                        }
                    };

                    std::vector<std::thread> threads;
                    for (unsigned int t = 0; t < std::thread::hardware_concurrency(); ++t)
                        threads.emplace_back(thread_func);
                    for (auto& th : threads) th.join();
                });
            }

            void basic_size_inference() {
                inference<Dims<1024, 16>>();
            }

            void non_pow_2_input_inference() {
                inference<Dims<1024 - 4, 16>>();
                inference<Dims<1024 + 4 * 32, 16>>();
            }

            void non_pow_2_output_inference() {
                inference<Dims<1024, 16 - 1>>();
                inference<Dims<1024, 16 + 1>>();
            }

            void small_input_inference() {
                inference<Dims<4, 16>>();
            }

            void small_output_inference() {
                inference<Dims<1024, 1>>();
            }

            void weird_sizes_inference() {
                inference<Dims<4, 1>>();
                inference<Dims<12, 3>>();
            }

            void bounds_checking() {
                constexpr size_t in  = 2048;
                constexpr size_t out = 16;

                using types = Types<int8_t, int32_t>;
                using dims  = Dims<in, out>;

                using scalar_params     = ParamComb_t<Scalar, types, dims>;
                using reference_layer_t = an::Layer<std::tuple_element_t<0, scalar_params>>;
                static reference_layer_t reference_layer{};

                static hwy::AlignedVector<int8_t>  weights(in * out);
                static hwy::AlignedVector<int32_t> biases(out);
                static hwy::AlignedVector<int8_t>  input(in);
                static hwy::AlignedVector<int32_t> output(out);

                weights.resize(in * out + 1);
                EXPECT_DEATH(reference_layer.load_weights(weights, biases), ".*");
                weights.resize(in * out);
                biases.resize(out + 1);
                EXPECT_DEATH(reference_layer.load_weights(weights, biases), ".*");
                biases.resize(out);
                reference_layer.load_weights(weights, biases);
                input.resize(in + 1);
                EXPECT_DEATH(reference_layer.forward(input, output), ".*");
                input.resize(in);
                output.resize(out + 1);
                EXPECT_DEATH(reference_layer.forward(input, output), ".*");
                output.resize(out);
                reference_layer.forward(input, output);
            }
        } // namespace
    } // namespace HWY_NAMESPACE
} // namespace chepp::tests::nnue::layers::affine

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::tests::nnue::layers::affine {
    namespace {
        using namespace chepp::nnue::layers;
        HWY_BEFORE_TEST(AffineTest);
        HWY_EXPORT_AND_TEST_P(AffineTest, basic_size_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, non_pow_2_input_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, non_pow_2_output_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, small_input_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, small_output_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, weird_sizes_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, bounds_checking);
    } // namespace
} // namespace chepp::tests::nnue::layers::affine
HWY_TEST_MAIN();
#endif // HWY_ONCE