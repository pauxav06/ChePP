#include "meta.h"
#include "utils.h"

#include <hwy/base.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "accum_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#include "accumulator-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::tests::nnue::layers::accumulator {
    namespace HWY_NAMESPACE {
        namespace {
            namespace hn = hwy::HWY_NAMESPACE;
            namespace an = chepp::nnue::layers::accumulator::HWY_NAMESPACE;

            using namespace hn;
            using namespace chepp::nnue::meta;
            using namespace chepp::nnue::layers;
            using namespace chepp::nnue::layers::accumulator;

            void accum() {
                constexpr size_t in         = 16000;
                constexpr size_t out        = 8;
                constexpr size_t max_active = 32;
                constexpr size_t Iterations = 10000;

                hwy::AlignedVector<int16_t> input(max_active);
                hwy::AlignedVector<int16_t> weights(in * out);
                hwy::AlignedVector<int16_t> biases(out);
                hwy::AlignedVector<int16_t> ref_output(out, 0);
                hwy::AlignedVector<int16_t> simd_output(out, 0);

                utils::fill_random(weights);
                utils::fill_random(biases);
                utils::fill_random(input, static_cast<int16_t>(0), static_cast<int16_t>(in));

                using types = Types<int16_t, int16_t>;
                using dims  = Dims<in, out>;

                using scalar_params     = ParamComb_t<Scalar, types, dims>;
                using reference_layer_t = an::Layer<std::tuple_element_t<0, scalar_params>>;
                reference_layer_t reference_layer;

                using UnrollOptions = std::tuple<Unroll<1>, Unroll<4>, Unroll<8>, Unroll<16>>;
                using simd_ops      = ParamComb_t<Simd, types, dims, UnrollOptions>;

                reference_layer.load_weights(weights.data(), biases.data());
                reference_layer.forward(input.data(), input.size(), ref_output.data());

                for_each_type<simd_ops>([&]<typename Opt>() {
                    an::Layer<Opt> layer;
                    layer.load_weights(weights.data(), biases.data());
                    layer.forward(input.data(), input.size(), simd_output.data());
                    HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), simd_output.size());
                });
            }
        } // namespace
    } // namespace HWY_NAMESPACE
} // namespace chepp::tests::nnue::layers::accumulator

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::tests::nnue::layers::accumulator {
    namespace {
        HWY_BEFORE_TEST(AccumTest);
        HWY_EXPORT_AND_TEST_P(AccumTest, accum);
    } // namespace
} // namespace chepp::tests::nnue::layers::accumulator
HWY_TEST_MAIN();
#endif // HWY_ONCE