#include <hwy/base.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "affine_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#include "../affine.h"

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
                    register_kernel<operation_t, default_config>();
                    register_kernel<operation_t, AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<operation_t, AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<operation_t, AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<operation_t, AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{1, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{2, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{4, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{8, AffineOperation::SumOfMulQuadAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{1, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{2, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{4, AffineOperation::MulPairwiseAdd}>();
                    register_kernel<operation_t, AffineSimdRowMaj{8, AffineOperation::MulPairwiseAdd}>();
                }

                template <typename Layer, auto cfg>
                void
                register_kernel() {
                    registery.register_kernel<Layer, HWY_TARGET, cfg>();
                }

                void
                inference() {
                    using namespace hwy;

                    AlignedVector<weights_type> weights(IS * OS);
                    AlignedVector<output_type>  biases(OS);

                    fill_random(weights);
                    fill_random(biases, 0, 128);

                    const layer_t layer{weights, biases};
                    const auto    kernels = registery.make_all_kernels<operation_t>();
                    const auto    ref     = registery.make_kernel<operation_t>(HWY_TARGET, default_config);
                    const auto    ref_ex  = ref->compile(layer);

                    for (const auto& kernel : kernels) {
                        const auto                 ex = kernel->compile(layer);
                        AlignedVector<input_type>  input(IS + kernel->input_padding());
                        AlignedVector<output_type> output(OS + kernel->output_padding());

                        fill_random(input, 0, 128);

                        FuncInput inp{};
                        Result    res{};
                        Params    params{};
                        params.verbose        = false;
                        params.target_rel_mad = 0.5;
                        hwy::MeasureClosure(
                            [&](auto) -> FuncOutput {
                                ex->forward(std::data(input), std::data(output));
                                return output[0];
                            },
                            &inp,
                            1,
                            &res,
                            params);

                        std::cout << TargetName(HWY_TARGET) << ": " << res.ticks << std::endl;

                        ref_ex->forward(std::data(input), std::data(output));
                        auto ref_output = output;
                        ex->forward(std::data(input), std::data(output));

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
// HWY_TEST_MAIN();
#endif // HWY_ONCE