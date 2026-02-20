#include <hwy/base.h>
#include <hwy/nanobenchmark.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "relu_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"
#include "relu-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace {
            namespace hn = hwy::HWY_NAMESPACE;

            template <typename InT, std::size_t IS, typename OutT, std::size_t Q>
            struct Test {
                using input_type  = InT;
                using output_type = OutT;
                using layer_t     = ClippedReLULayer<input_type, IS, output_type, Q>;

                Test() {
                    register_kernel<layer_t, 0>();
                    register_kernel<layer_t, ClippedReluSimd{1}>();
                    register_kernel<layer_t, ClippedReluSimd{2}>();
                    register_kernel<layer_t, ClippedReluSimd{4}>();
                    register_kernel<layer_t, ClippedReluSimd{8}>();
                    register_kernel<layer_t, ClippedReluSimd{16}>();
                }

                template <typename Layer, auto cfg>
                void
                register_kernel() {
                    auto key = registery.register_kernel<Layer, HWY_TARGET, cfg>();
                    if (!cfg_mapping.contains(key)) {
                        cfg_mapping.emplace(key, cfg_mapping.size());
                    }
                }

                void
                inference() {
                    using namespace hwy;

                    const auto layer   = std::make_shared<layer_t>();
                    const auto kernels = registery.make_all_kernels(layer);
                    const auto ref     = *registery.make_kernel(layer, HWY_TARGET, 0);

                    for (const auto& kernel : kernels) {
                        AlignedVector<input_type>  input(IS + kernel->input_padding());
                        AlignedVector<output_type> output(IS + kernel->output_padding());

                        fill_random(input, -10, 10);

                        FuncInput   inp{};
                        hwy::Result res{};
                        Params      params{};
                        params.verbose        = false;
                        params.target_rel_mad = 0.5;
                        hwy::MeasureClosure(
                            [&](auto) -> FuncOutput {
                                kernel->forward(std::data(input), std::data(output));
                                return output[0];
                            },
                            &inp,
                            1,
                            &res,
                            params);

                        std::cout << TargetName(HWY_TARGET) << ": " << res.ticks << std::endl;

                        ref->forward(std::data(input), std::data(output));
                        auto ref_output = output;
                        fill_random(output);
                        kernel->forward(std::data(input), std::data(output));

                        HWY_ASSERT_ARRAY_EQ(ref_output.data(), output.data(), IS);
                    }
                }

                KernelRegistry                             registery{};
                std::unordered_map<KernelKey, std::size_t> cfg_mapping{};
            };

            void
            runAllTests() {
                // Important
                Test<int16_t, 2048, uint8_t, 1>{}.inference();
                Test<int32_t, 16, uint8_t, 64>{}.inference();
                Test<int32_t, 32, uint8_t, 64>{}.inference();

                // Edge cases
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