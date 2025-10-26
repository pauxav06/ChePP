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

#include "affine-inl.h"
#include "network-inl.h"
;

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers::affine {
    namespace HWY_NAMESPACE {
        namespace {

            namespace hn = hwy::HWY_NAMESPACE;

            using namespace hn;
            using namespace chepp::nnue::meta;
            using namespace chepp::nnue::layers;

            void
            inference(Dims dims) {
                using namespace hwy;

                const size_t input_size  = dims.in;
                const size_t output_size = dims.out;
                using input_type         = int8_t;
                using output_type        = int32_t;

                AlignedVector<input_type>  weights(input_size * output_size);
                AlignedVector<output_type> biases(output_size);

                chepp::utils::fill_random(weights);
                chepp::utils::fill_random(biases, 0, 0);

                using types   = Types<input_type, output_type>;
                using layer_t = Layer<types>;
                layer_t::Params params{
                    .dims    = dims,
                    .weights = weights,
                    .biases  = biases,
                };

                auto layer = std::make_shared<layer_t>(params);
                auto ref   = make_state(layer, Scalar{});

                ref->init();

                using UnrollOptions = std::tuple<Unroll<1>, Unroll<8>>;
                using DotOps        = std::tuple<SumOfMulQuadAcc, SumOfMulPairAcc>;
                using simd_opts     = Cartesian_t<std::tuple<SimdColMaj>, UnrollOptions, DotOps>;

                constexpr int rep = 10000;

                for_each_type<simd_opts>([&]<typename Opt>() {
                    auto simd = make_state(layer, Opt{});
                    simd->init();

                    const auto padded_input_size  = simd->input_buffer_constraints().get(SizeConstraint{}).min_size;
                    const auto padded_output_size = simd->output_buffer_constraints().get(SizeConstraint{}).min_size;
                    std::cout << input_size << " " << padded_input_size << std::endl;
                    AlignedVector<input_type>  input(utils::pad_up(input_size, padded_input_size));
                    AlignedVector<output_type> ref_output(utils::pad_up(output_size, padded_output_size));

                    chepp::utils::fill_random(input, 0);

                    ref->forward(input, ref_output);

                    auto thread_func = [&]() {
                        AlignedVector<int32_t> simd_output(utils::pad_up(output_size, padded_output_size));
                        for (int i = 0; i < rep; ++i) {
                            simd->forward(input, simd_output);
                            HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), output_size);
                        }
                    };

                    std::vector<std::thread> threads;
                    for (unsigned int t = 0; t < std::thread::hardware_concurrency(); ++t)
                        threads.emplace_back(thread_func);
                    for (auto& th : threads) th.join();
                });
            }

            void
            basic_size_inference() {
                inference(Dims{1024, 16});
            }

            void
            non_pow_2_input_inference() {
                inference(Dims{1024 - 4, 16});
                inference(Dims{1024 + 4 * 32, 16});
            }

            void
            non_pow_2_output_inference() {
                inference(Dims{1024, 16 - 1});
                inference(Dims{1024, 16 + 1});
            }

            void
            small_input_inference() {
                inference(Dims{4, 16});
            }

            void
            small_output_inference() {
                inference(Dims{1024, 1});
            }

            void
            non_mult_4_input_inference() {
                inference(Dims{1, 16});
                inference(Dims{14, 16});
            }

            void
            weird_sizes_inference() {
                inference(Dims{4, 1});
                inference(Dims{12, 3});
                inference(Dims{3, 8});
                inference(Dims{309, 678});

                auto layer = network::l1::get_layer(0);
                auto state = chepp::nnue::network::l1::HWY_NAMESPACE::make_state(0, layer);
                state->init();
                AlignedVector<int8_t>  input(2048);
                AlignedVector<int32_t> output(16);
                state->forward(input, output);
            }

            /**
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
            **/
        } // namespace
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::affine

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue::layers::affine {
    namespace {
        HWY_BEFORE_TEST(AffineTest);
        HWY_EXPORT_AND_TEST_P(AffineTest, basic_size_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, non_pow_2_input_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, non_pow_2_output_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, small_input_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, small_output_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, non_mult_4_input_inference);
        HWY_EXPORT_AND_TEST_P(AffineTest, weird_sizes_inference);
        // HWY_EXPORT_AND_TEST_P(AffineTest, bounds_checking);
        HWY_AFTER_TEST();
    } // namespace
} // namespace chepp::nnue::layers::affine
// HWY_TEST_MAIN();
#endif // HWY_ONCE