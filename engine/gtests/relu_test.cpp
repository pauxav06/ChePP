#include "meta.h"
#include "utils.h"

#include <hwy/base.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "relu_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/print.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#include "relu-inl.h"
// #include "network-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers::relu {
    namespace HWY_NAMESPACE {
        namespace {
            namespace hn = hwy::HWY_NAMESPACE;
            using namespace hn;

            template <typename InT, typename OutT, size_t S>
            void
            inference(const Dims dims) {
                using types   = Types<InT, OutT>;
                using shift   = Shift<S>;
                using layer_t = Layer<types, shift>;

                typename layer_t::Params params{.dims = dims};

                auto layer = std::make_shared<layer_t>(params);
                auto ref   = make_state(layer, Scalar{});
                ref->init();

                using unrolls      = std::tuple<Unroll<1>, Unroll<2>, Unroll<4>>;
                using simd_options = meta::Cartesian_t<std::tuple<Simd>, unrolls>;

                meta::for_each_type<simd_options>([&]<typename Opt>() {
                    auto simd = make_state(layer, Opt{});
                    simd->init();

                    size_t input_size =
                        utils::pad_up(dims.in, simd->input_buffer_constraints().get(SizeConstraint()).min_size);
                    size_t output_size =
                        utils::pad_up(dims.out, simd->output_buffer_constraints().get(SizeConstraint()).min_size);
                    hwy::AlignedVector<InT>  input(input_size);
                    hwy::AlignedVector<OutT> ref_output(output_size);
                    hwy::AlignedVector<OutT> simd_output(output_size);
                    chepp::utils::fill_random(input);

                    ref->forward(input, ref_output);
                    simd->forward(input, simd_output);

                    HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), dims.in);
                });
            }

            void
            inference_single_narrow() {
                inference<int32_t, int16_t, 0>(Dims{1024, 1024});
                inference<int16_t, int8_t, 0>(Dims{8192, 8192});
            }

            void
            inference_single_narrow_shift() {
                inference<int32_t, int16_t, 6>(Dims{8192, 8192});
                inference<int16_t, int8_t, 6>(Dims{8192, 8192});
                inference<int32_t, int16_t, 15>(Dims{8192, 8192});
                inference<int16_t, int8_t, 15>(Dims{8192, 8192});
            }

            void
            inference_x2_narrow() {
                inference<int32_t, int8_t, 0>(Dims{8192, 8192});
            }

            void
            inference_x2_narrow_shift() {
                inference<int32_t, int8_t, 6>(Dims{8192, 8192});
                inference<int32_t, int8_t, 15>(Dims{8192, 8192});
            }

            void
            inference_weird_sizes() {
                inference<int32_t, int8_t, 6>(Dims{304, 304});
                inference<int32_t, int8_t, 15>(Dims{2, 2});
                inference<int32_t, int8_t, 15>(Dims{11, 11});
                inference<int32_t, int8_t, 15>(Dims{333, 333});
            }
            /**
            void bounds_checking() {
                using InT                     = int16_t;
                using OutT                    = int32_t;
                static constexpr size_t size  = 1024;
                static constexpr size_t shift = 8;

                hwy::AlignedVector<InT>  input(size);
                hwy::AlignedVector<OutT> output(size, 0);

                using types = Types<InT, OutT>;
                using dims  = Dims<size, size>;

                using shifts            = std::tuple<std::integral_constant<size_t, shift>>;
                using ref_params        = ParamComb_t<Scalar, types, dims, shifts>;
                using reference_layer_t = an::Layer<std::tuple_element_t<0, ref_params>>;
                reference_layer_t reference_layer;

                reference_layer.forward(input, output);
                output.resize(size + 1);
                EXPECT_DEATH(reference_layer.forward(input, output), ".*");
                output.resize(size);
                input.resize(size + 1);
                EXPECT_DEATH(reference_layer.forward(input, output), ".*");
            }
            **/
        } // namespace
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers::relu

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue::layers::relu {
    namespace {
        HWY_BEFORE_TEST(ReluTest);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_single_narrow);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_single_narrow_shift);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_x2_narrow);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_x2_narrow_shift);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_weird_sizes);
        // HWY_EXPORT_AND_TEST_P(ReluTest, bounds_checking);
        HWY_AFTER_TEST();
    } // namespace
} // namespace chepp::nnue::layers::relu
HWY_TEST_MAIN();
#endif // HWY_ONCE