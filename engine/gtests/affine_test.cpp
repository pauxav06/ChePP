#ifndef AFFINE_TEST_H
#define AFFINE_TEST_H
#include <random>
#include <stdint.h>

namespace chepp::nnue::layers
{
    namespace affine
    {
        namespace
        {

            struct Param
            {
                size_t InSz{};
                size_t OutSz{};
            };

        }
    }
    namespace relu
    {
        namespace
        {
        }
    }
}

#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "affine_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"
#include <hwy/base.h>

#include "affine-inl.h"
#include "relu-inl.h"

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::layers
{
    namespace affine
    {
        namespace HWY_NAMESPACE
        {
            namespace
            {

                template <typename T>
                void fill_random(std::vector<T>& v, int min, int max, unsigned seed = 1234)
                {
                    std::mt19937                       rng(seed);
                    std::uniform_int_distribution<int> dist(min, max);
                    for (auto& x : v)
                        x = static_cast<T>(dist(rng));
                }

                void reference_affine(const std::vector<int8_t>& input, const std::vector<int8_t>& weights,
                                      const std::vector<int32_t>& biases, size_t Rows, size_t Cols,
                                      std::vector<int32_t>& out)
                {
                    out.assign(Rows, 0);
                    for (size_t r = 0; r < Rows; r++)
                    {
                        int32_t acc = biases[r];
                        for (size_t c = 0; c < Cols; c++)
                        {
                            acc += static_cast<int32_t>(input[c]) * static_cast<int32_t>(weights[r * Cols + c]);
                        }
                        out[r] = acc;
                    }
                }

                namespace hn = hwy::HWY_NAMESPACE;

                template <Param P>
                void affine()
                {
                    std::cout << "Test passed for target: " << hwy::TargetName(HWY_TARGET) << std::endl;
                    constexpr size_t Rows       = P.OutSz;
                    constexpr size_t Cols       = P.InSz;
                    constexpr size_t Iterations = 10000;

                    std::vector<int8_t>  weights(Rows * Cols);
                    std::vector<int32_t> biases(Rows);
                    std::vector<int8_t>  input(Cols);
                    std::vector<int32_t> ref_output(Rows);
                    std::vector<int32_t> simd_output(Rows);

                    fill_random(weights, -3, 3);
                    fill_random(biases, -10, 10);
                    fill_random(input, 0, 127);

                    using namespace std::chrono;
                    int64_t total_ref_duration = 0;
                    auto start_ref = high_resolution_clock::now();
                    for (size_t iter = 0; iter < Iterations; ++iter)
                    {
                        reference_affine(input, weights, biases, Rows, Cols, ref_output);
                    }
                    auto end_ref = high_resolution_clock::now();
                    total_ref_duration += duration_cast<nanoseconds>(end_ref - start_ref).count();

                    constexpr VNNIKernelParams params{8, true, true};
                    using layer_t = VNNIKernel<Rows, Cols, params>;
                    layer_t layer;
                    layer.load_weights(weights.data(), biases.data());

                    int64_t total_simd_duration = 0;
                    auto start_simd = high_resolution_clock::now();
                    for (size_t iter = 0; iter < Iterations; ++iter)
                    {
                        layer.forward(input.data(), simd_output.data());
                    }
                    auto end_simd = high_resolution_clock::now();
                    total_simd_duration += duration_cast<nanoseconds>(end_simd - start_simd).count();

                    reference_affine(input, weights, biases, Rows, Cols, ref_output);
                    HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), simd_output.size());
                    std::cout << "Reference (naive) affine avg: " << total_ref_duration / Iterations << " ns\n";
                    std::cout << "SIMD affine layer avg:        " << total_simd_duration / Iterations << " ns\n";
                }

                void affine_all()
                {
                    constexpr Param p = {2048, 16};
                    affine<p>();
                }
            }
        }
    }

    namespace relu
    {
        namespace HWY_NAMESPACE
        {
            namespace
            {
                namespace hn = hwy::HWY_NAMESPACE;

                template <typename T>
                int8_t clip_relu_ref(T x)
                {
                    if (x < 0)
                        return 0;
                    if (x > 127)
                        return 127;
                    return static_cast<int8_t>(x);
                }

                void relu_test()
                {
                    using namespace std::chrono;

                    constexpr size_t N = 2048 + 1;
                    constexpr size_t Iterations = 100000;

                    using InT = int32_t;

                    std::vector<InT, hwy::AlignedAllocator<InT>> input(N);
                    std::vector<int8_t, hwy::AlignedAllocator<int8_t>>  output(N);
                    std::vector<int8_t, hwy::AlignedAllocator<int8_t>>  ref(N);

                    std::mt19937                           rng(1234);
                    std::uniform_int_distribution dist(std::numeric_limits<InT>::min(),
                                                                std::numeric_limits<InT>::max());
                    for (auto& x : input)
                        x = dist(rng);

                    constexpr ClippedReLUParams<InT> params{0, 127, 0, 8};

                    int64_t total_simd_duration = 0;
                    auto start_simd = high_resolution_clock::now();

                    for (size_t iter = 0; iter < Iterations; ++iter)
                    {
                        SIMDNarrowingX2ClippedReLU<N, InT, params>::forward(input.data(), output.data());

                    }
                    auto end_simd = high_resolution_clock::now();
                    total_simd_duration += duration_cast<nanoseconds>(end_simd - start_simd).count();

                    int64_t total_ref_duration = 0;
                    auto start_ref = high_resolution_clock::now();

                    for (size_t iter = 0; iter < Iterations; ++iter)
                    {
                        for (size_t i = 0; i < N; i++)
                            ref[i] = clip_relu_ref(input[i]);

                    }
                    auto end_ref = high_resolution_clock::now();
                    total_ref_duration += duration_cast<nanoseconds>(end_ref - start_ref).count();
                    for (size_t i = 0; i < N; i++)
                    {
                        EXPECT_EQ(output[i], ref[i]) << "Mismatch at index " << i << " input=" << input[i];
                    }

                    std::cout << "Test passed for target: " << hwy::TargetName(HWY_TARGET) << std::endl;
                    std::cout << "Reference (scalar) ReLU avg: " << total_ref_duration / Iterations << " ns\n";
                    std::cout << "SIMD ReLU layer avg:         " << total_simd_duration / Iterations << " ns\n";
                }

            }
        }

    } // namespace
    // NOLINTNEXTLINE(google-readability-namespace-comments)
} // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue::layers
{
    namespace affine
    {
        namespace
        {
            HWY_BEFORE_TEST(AffineTest);
            HWY_EXPORT_AND_TEST_P(AffineTest, affine_all);
        }
    }
    namespace relu
    {
        namespace
        {
            HWY_AFTER_TEST();
            HWY_BEFORE_TEST(ReluTest);
            HWY_EXPORT_AND_TEST_P(ReluTest, relu_test);
            HWY_AFTER_TEST();
        }
    }
}
HWY_TEST_MAIN();
#endif // HWY_ONCE