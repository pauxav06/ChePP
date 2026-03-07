#include <hwy/base.h>
#include <hwy/nanobenchmark.h>

#undef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "accum_test.cpp"
#include <hwy/foreach_target.h>

#include <hwy/highway.h>
#include <hwy/tests/test_util-inl.h>

#include "../accumulator-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace {
            namespace hn = hwy::HWY_NAMESPACE;

            template <typename IT, std::size_t IS, typename T, std::size_t OS>
            struct Test {
                using type        = T;
                using idx_t       = IT;
                using operation_t = Accumulator<idx_t, IS, type, OS>;
                using layer_t     = operation_t::layer_t;

                Test() {
                    register_kernel<default_config>();
                    register_kernel<AccumulatorSimd{1}>();
                    register_kernel<AccumulatorSimd{2}>();
                    register_kernel<AccumulatorSimd{4}>();
                    register_kernel<AccumulatorSimd{8}>();
                    register_kernel<AccumulatorSimd{16}>();
                }

                template <auto cfg>
                void
                register_kernel() {
                    registery.register_kernel<operation_t, HWY_TARGET, cfg>();
                }

                void
                inference() {
                    using namespace hwy;

                    std::vector<type> weights(IS * OS);
                    fill_random(weights);
                    std::vector<type> biases(OS);
                    fill_random(biases);

                    auto       layer   = std::make_shared<layer_t>(weights, biases);
                    const auto kernels = registery.make_all_kernels<operation_t>(layer);
                    auto       ref     = registery.make_kernel<operation_t>(layer, HWY_TARGET, default_config);

                    for (const auto& kernel : kernels) {
                        AlignedVector<type> output(operation_t::output_size_v + kernel->padding());

                        std::vector<idx_t> idx(IS);
                        std::iota(std::begin(idx), std::end(idx), static_cast<idx_t>(0));

                        ref->forward(std::data(idx), std::size(idx), std::data(output));
                        auto ref_output = output;
                        fill_random(output);
                        kernel->forward(std::data(idx), std::size(idx), std::data(output));

                        HWY_ASSERT_ARRAY_EQ(ref_output.data(), output.data(), OS);
                    }
                }

                KernelRegistry registery{};
            };

            void
            runAllTests() {
                // Important
                Test<uint16_t, 1024, int16_t, 1024>{}.inference();
                Test<uint16_t, 1024, int16_t, 8>{}.inference();

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
HWY_TEST_MAIN();
// HWY_TEST_MAIN();
#endif // HWY_ONCE