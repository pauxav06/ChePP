#include <hwy/base.h>
#include <hwy/nanobenchmark.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "accum_test.cpp"
#include "hwy/foreach_target.h"

#include "accumulator-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace {
            namespace hn = hwy::HWY_NAMESPACE;

            template <typename IT, std::size_t IS, typename T, std::size_t OS>
            struct Test {
                using type    = T;
                using idx_t   = IT;
                using layer_t = AccumulatorLayer<idx_t, IS, type, OS>;

                Test() {
                    register_kernel<layer_t, 0>();
                    register_kernel<layer_t, AccumulatorSimd{1}>();
                    register_kernel<layer_t, AccumulatorSimd{2}>();
                    register_kernel<layer_t, AccumulatorSimd{4}>();
                    register_kernel<layer_t, AccumulatorSimd{8}>();
                    register_kernel<layer_t, AccumulatorSimd{16}>();
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

                    std::vector<type> weights(IS * OS);
                    std::vector<type> biases(OS);

                    const auto layer   = std::make_shared<layer_t>(weights, biases);
                    const auto kernels = registery.make_all_kernels(layer);
                    const auto ref     = *registery.make_kernel(layer, HWY_TARGET, 0);

                    for (const auto& kernel : kernels) {
                        AlignedVector<type> output(OS + kernel->padding());

                        std::vector<idx_t> idx(IS);
                        std::ranges::iota(idx, 0);

                        FuncInput   inp{};
                        hwy::Result res{};
                        Params      params{};
                        params.verbose        = false;
                        params.target_rel_mad = 0.5;
                        hwy::MeasureClosure(
                            [&](auto) -> FuncOutput {
                                kernel->forward(std::data(idx), std::size(idx), std::data(output));
                                return output[0];
                            },
                            &inp,
                            1,
                            &res,
                            params);

                        std::cout << TargetName(HWY_TARGET) << ": " << res.ticks << std::endl;

                        ref->forward(std::data(idx), std::size(idx), std::data(output));
                        auto ref_output = output;
                        fill_random(output);
                        kernel->forward(std::data(idx), std::size(idx), std::data(output));

                        HWY_ASSERT_ARRAY_EQ(ref_output.data(), output.data(), OS);
                    }
                }

                KernelRegistry                             registery{};
                std::unordered_map<KernelKey, std::size_t> cfg_mapping{};
            };

            void
            runAllTests() {
                // Important
                // Test<uint16_t, 2048, int16_t, 1>{}.inference();
                Test<uint16_t, 16, int16_t, 64>{}.inference();
                Test<uint16_t, 32, int16_t, 64>{}.inference();

                // Edge cases
            }

        } // namespace
    }     // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue::layers {
    namespace {
        HWY_BEFORE_TEST(AccumTest);
        HWY_EXPORT_AND_TEST_P(AccumTest, runAllTests);
        HWY_AFTER_TEST();
    } // namespace
} // namespace chepp::nnue::layers
// HWY_TEST_MAIN();
#endif // HWY_ONCE