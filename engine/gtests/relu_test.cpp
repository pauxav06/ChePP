#include <hwy/base.h>
#include <hwy/nanobenchmark.h>

#undef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "relu_test.cpp"
#include "hwy/foreach_target.h"

#include <hwy/highway.h>
#include <hwy/tests/test_util-inl.h>

#include "../relu-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace {
            namespace hn = hwy::HWY_NAMESPACE;

            template <typename InT, std::size_t IS, typename OutT, std::size_t Q>
            struct Test {
                using input_type  = InT;
                using output_type = OutT;
                using operation_t = ClippedRelu<input_type, IS, output_type, Q>;
                using layer_t     = operation_t::layer_t;

                Test() {
                    register_kernel<default_config>();
                    register_kernel<ClippedReluSimd{1}>();
                    register_kernel<ClippedReluSimd{2}>();
                    register_kernel<ClippedReluSimd{4}>();
                    register_kernel<ClippedReluSimd{8}>();
                    // register_kernel<layer_t, ClippedReluSimd{16}>();
                }

                template <auto cfg>
                void
                register_kernel() {
                    registery.register_kernel<operation_t, HWY_TARGET, cfg>();
                }

                void
                inference() {
                    using namespace hwy;

                    const auto layer   = std::make_shared<layer_t>();
                    const auto kernels = registery.make_all_kernels<operation_t>(layer);
                    const auto ref     = registery.make_kernel<operation_t>(layer, HWY_TARGET, default_config);

                    for (const auto& kernel : kernels) {
                        AlignedVector<input_type>  input(IS + kernel->input_padding());
                        AlignedVector<output_type> output(IS + kernel->output_padding());

                        fill_random(input, -10, 10);

                        ref->forward(std::data(input), std::data(output));
                        auto ref_output = output;
                        fill_random(output);
                        kernel->forward(std::data(input), std::data(output));

                        HWY_ASSERT_ARRAY_EQ(ref_output.data(), output.data(), IS);
                    }
                }

                KernelRegistry registery{};
            };

            void
            runAllTests() {
                // Important
                Test<int16_t, 2048, uint8_t, 1>{}.inference();
                Test<int32_t, 16, uint8_t, 64>{}.inference();
                Test<int32_t, 32, uint8_t, 64>{}.inference();
                // Edge cases
                Test<int32_t, 1, uint8_t, 4>{}.inference();
                Test<int32_t, 23, uint8_t, 8>{}.inference();
                Test<int32_t, 765, uint8_t, 16>{}.inference();
            }

        } // namespace
    }     // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue::layers {
    namespace {
        HWY_BEFORE_TEST(ReluTest);
        HWY_EXPORT_AND_TEST_P(ReluTest, runAllTests);
        HWY_AFTER_TEST();
    } // namespace
} // namespace chepp::nnue::layers
// HWY_TEST_MAIN();
#endif // HWY_ONCE