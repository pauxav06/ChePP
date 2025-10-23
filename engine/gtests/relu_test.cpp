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

namespace chepp::tests::nnue::layers::relu {
    namespace HWY_NAMESPACE {
        namespace {
            namespace hn = hwy::HWY_NAMESPACE;
            namespace an = chepp::nnue::layers::relu::HWY_NAMESPACE;

            using namespace hn;
            using namespace chepp::nnue::meta;
            using namespace chepp::nnue::layers;
            using namespace chepp::nnue::layers::relu;

            template <typename InT, typename OutT, size_t Size, size_t Shift>
            void inference() {

                hwy::AlignedVector<InT>  input(Size);
                hwy::AlignedVector<OutT> ref_output(Size, 0);
                hwy::AlignedVector<OutT> simd_output(Size, 0);

                utils::fill_random(input);

                using types = Types<InT, OutT>;
                using dims  = Dims<Size, Size>;

                using shifts            = std::tuple<std::integral_constant<size_t, Shift>>;
                using ref_params        = ParamComb_t<Scalar, types, dims, shifts>;
                using reference_layer_t = an::Layer<std::tuple_element_t<0, ref_params>>;
                reference_layer_t reference_layer;

                using simd_ops = ParamComb_t<Simd, types, dims, shifts>;

                reference_layer.forward(input, ref_output);

                for_each_type<simd_ops>([&]<typename Opt>() {
                    an::Layer<Opt> layer;
                    layer.forward(input, simd_output);
                    HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), simd_output.size());
                });
            }

            void inference_single_narrow() {
                inference<int32_t, int16_t, 1024, 0>();
                inference<int16_t, int8_t, 8192, 0>();
            }

            void inference_single_narrow_shift() {
                inference<int32_t, int16_t, 8192, 6>();
                inference<int16_t, int8_t, 8192, 6>();
                inference<int32_t, int16_t, 8192, 15>();
                inference<int16_t, int8_t, 8192, 15>();
            }

            void inference_x2_narrow() {
                inference<int32_t, int8_t, 8192, 0>();
            }

            void inference_x2_narrow_shift() {
                inference<int32_t, int8_t, 8192, 6>();
                inference<int32_t, int8_t, 8192, 15>();
            }

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
        } // namespace
    } // namespace HWY_NAMESPACE
} // namespace chepp::tests::nnue::layers::relu

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::tests::nnue::layers::relu {
    namespace {
        HWY_BEFORE_TEST(ReluTest);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_single_narrow);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_single_narrow_shift);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_x2_narrow);
        HWY_EXPORT_AND_TEST_P(ReluTest, inference_x2_narrow_shift);
        HWY_EXPORT_AND_TEST_P(ReluTest, bounds_checking);
        HWY_AFTER_TEST();
    } // namespace
} // namespace chepp::tests::nnue::layers::relu
HWY_TEST_MAIN();
#endif // HWY_ONCE