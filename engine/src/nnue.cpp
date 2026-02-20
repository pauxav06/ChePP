#include "nnue.h"
#include <hwy/base.h>
#include <hwy/nanobenchmark.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "nnue.cpp"

#include "hwy/foreach_target.h"

#include "accumulator-inl.h"
#include "affine-inl.h"
#include "hwy/highway.h"
#include "relu-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace {} // namespace
    }                // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue {

    void
    Arch::register_kernels() {
        registry.register_kernel<accum_t, HWY_STATIC_TARGET, default_config>();

        registry.register_kernel<accum_t, HWY_STATIC_TARGET, AccumulatorSimd{1}>();
        registry.register_kernel<accum_t, HWY_STATIC_TARGET, AccumulatorSimd{2}>();
        registry.register_kernel<accum_t, HWY_STATIC_TARGET, AccumulatorSimd{4}>();
        registry.register_kernel<accum_t, HWY_STATIC_TARGET, AccumulatorSimd{8}>();
        registry.register_kernel<accum_t, HWY_STATIC_TARGET, AccumulatorSimd{16}>();

        registry.register_kernel<psqt_t, HWY_STATIC_TARGET, default_config>();

        registry.register_kernel<act0_t, HWY_STATIC_TARGET, default_config>();
        registry.register_kernel<act0_t, HWY_STATIC_TARGET, ClippedReluSimd{1}>();
        registry.register_kernel<act0_t, HWY_STATIC_TARGET, ClippedReluSimd{2}>();
        registry.register_kernel<act0_t, HWY_STATIC_TARGET, ClippedReluSimd{4}>();
        registry.register_kernel<act0_t, HWY_STATIC_TARGET, ClippedReluSimd{8}>();
        registry.register_kernel<act0_t, HWY_STATIC_TARGET, ClippedReluSimd{16}>();

        registry.register_kernel<l1_t, HWY_STATIC_TARGET, default_config>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{1, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{2, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{4, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{8, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{1, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{2, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{4, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdRowMaj{8, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l1_t, HWY_STATIC_TARGET, AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>();

        registry.register_kernel<act1_t, HWY_STATIC_TARGET, default_config>();

        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{1, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{2, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{4, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{8, AffineOperation::SumOfMulQuadAdd}>();
        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{1, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{2, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{4, AffineOperation::MulPairwiseAdd}>();
        registry.register_kernel<l2_t, HWY_STATIC_TARGET, AffineSimdColMaj{8, AffineOperation::MulPairwiseAdd}>();

        registry.register_kernel<l2_t, HWY_STATIC_TARGET, default_config>();
        registry.register_kernel<act2_t, HWY_STATIC_TARGET, default_config>();
        registry.register_kernel<l3_t, HWY_STATIC_TARGET, default_config>();
    }
} // namespace chepp::nnue
// HWY_TEST_MAIN();
#endif // HWY_ONCE