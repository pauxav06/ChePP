#include <hwy/base.h>

#include "hwy/nanobenchmark.h"

#undef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "affine_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

#include "../affine-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace {

            namespace hn = hwy::HWY_NAMESPACE;

            using weights_type = int8_t;
            using input_type   = uint8_t;
            using output_type  = int32_t;
            using bias_type    = int32_t;

            template <std::size_t IS, std::size_t OS>
            struct Test {
                using operation_t = Affine<input_type, IS, output_type, OS, weights_type, bias_type>;
                using layer_t     = operation_t::layer_t;

                Test() {
                    register_kernel<default_config>();
                    register_kernel<AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<AffineSimdRowMaj{1, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdRowMaj{2, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdRowMaj{4, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdRowMaj{8, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<AffineSimdRowMaj{1, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<AffineSimdRowMaj{2, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<AffineSimdRowMaj{4, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<AffineSimdRowMaj{8, AffineOperation::MulPairwiseAdd}>();
                }

                template <auto cfg>
                void
                register_kernel() {
                    registery.register_kernel<operation_t, HWY_TARGET, cfg>();
                }

                void
                inference() {
                    using namespace hwy;

                    AlignedVector<weights_type> weights(IS * OS);
                    AlignedVector<output_type>  biases(OS);

                    fill_random(weights);
                    fill_random(biases, 0, 128);

                    const auto layer   = std::make_shared<layer_t>(weights, biases);
                    const auto kernels = registery.make_all_kernels<operation_t>(layer);
                    const auto ref     = registery.make_kernel<operation_t>(layer, HWY_TARGET, default_config);

                    for (const auto& kernel : kernels) {
                        AlignedVector<input_type>  input(IS + kernel->input_padding());
                        AlignedVector<output_type> output(OS + kernel->output_padding());

                        fill_random(input, 0, 128);

                        ref->forward(std::data(input), std::data(output));
                        auto ref_output = output;
                        kernel->forward(std::data(input), std::data(output));

                        HWY_ASSERT_ARRAY_EQ(ref_output.data(), output.data(), OS);
                    }
                }

                KernelRegistry registery{};
            };

            void
            runAllTests() {
                // Important
                Test<2048, 16>{}.inference();
                Test<16, 32>{}.inference();
                Test<32, 1>{}.inference();

                // Edge cases
                Test<1, 32>{}.inference();
                Test<1, 1>{}.inference();
                Test<11, 11>{}.inference();
            }

        } // namespace
    }     // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue::layers {
    namespace {
        HWY_BEFORE_TEST(AffineTest);
        HWY_EXPORT_AND_TEST_P(AffineTest, runAllTests);
        HWY_AFTER_TEST();
    } // namespace
} // namespace chepp::nnue::layers
HWY_TEST_MAIN();
#endif // HWY_ONCE