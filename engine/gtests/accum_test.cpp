#include <vector>
#include <random>
#include <cassert>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "accum_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"
#include <hwy/base.h>

#include "accumulator-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue
{
    namespace accum
    {
        namespace HWY_NAMESPACE
        {
            namespace
            {
                template <typename T, typename Alloc>
                void fill_random(std::vector<T, Alloc>& v, int min, int max, unsigned seed = 1234)
                {
                    std::mt19937 rng(seed);
                    std::uniform_int_distribution<int> dist(min, max);
                    for (auto& x : v) x = static_cast<T>(dist(rng));
                }

                template <typename T>
                void reference_accum(const T* vec,
                                     const T* weights,
                                     size_t Cols,
                                     const int16_t* indices,
                                     size_t n,
                                     T* out)
                {
                    for (size_t c = 0; c < Cols; ++c)
                    {
                        out[c] = vec[c];
                    }
                    for (size_t i = 0; i < n; ++i)
                    {
                        int16_t idx = indices[i];
                        for (size_t c = 0; c < Cols; ++c)
                        {
                            out[c] += weights[idx * Cols + c];
                        }
                    }
                }

                template <typename Kernel, size_t Rows, size_t Cols>
                void accumulator_test()
                {
                    std::cout << "Test passed for target: " << hwy::TargetName(HWY_TARGET) << std::endl;

                    constexpr size_t Iterations = 100000;

                    std::vector<int32_t, hwy::AlignedAllocator<int32_t>> weights(Rows * Cols);
                    std::vector<int32_t, hwy::AlignedAllocator<int32_t>> input(Cols, 0);
                    std::vector<int32_t, hwy::AlignedAllocator<int32_t>> ref_output(Cols);
                    std::vector<int32_t, hwy::AlignedAllocator<int32_t>> simd_output(Cols);
                    std::vector<int16_t, hwy::AlignedAllocator<int16_t>> indices = {0, Rows / 2, Rows - 1};

                    fill_random(weights, -10, 10);

                    Kernel kernel;
                    kernel.load_weights(weights.data());

                    using namespace std::chrono;
                    int64_t total_ref_duration = 0;
                    auto start = high_resolution_clock::now();

                    for (size_t iter = 0; iter < Iterations; ++iter)
                    {
                        reference_accum(input.data(), weights.data(), Cols, indices.data(), indices.size(), ref_output.data());
                    }
                    auto end = high_resolution_clock::now();
                    total_ref_duration += duration_cast<nanoseconds>(end - start).count();

                    int64_t total_kernel_duration = 0;
                    start = high_resolution_clock::now();
                    for (size_t iter = 0; iter < Iterations; ++iter)
                    {
                        //kernel.forward(input.data(), indices.data(), indices.size(), simd_output.data(), true);

                    }
                    end = high_resolution_clock::now();
                    total_kernel_duration += duration_cast<nanoseconds>(end - start).count();

                    for (size_t c = 0; c < Cols; ++c)
                        HWY_ASSERT_EQ(simd_output[c], ref_output[c]);

                    std::cout << "Reference accumulator avg: " << total_ref_duration / Iterations << " ns\n";
                    std::cout << "SIMD/Kernel accumulator avg: " << total_kernel_duration / Iterations << " ns\n";
                }

                void accumulator_all()
                {
                    using kernel_t = chepp::nnue::layers::accum::HWY_NAMESPACE::SIMDKernel<int32_t, 32, 1024, int16_t>;
                    accumulator_test<kernel_t, 32, 1024>();
                }
            }
        }
    }
}

HWY_AFTER_NAMESPACE();


#if HWY_ONCE

namespace chepp::nnue
{
    namespace accum
    {
        namespace HWY_NAMESPACE
        {
            namespace
            {
                HWY_BEFORE_TEST(AccumulatorTest);
                HWY_EXPORT_AND_TEST_P(AccumulatorTest, accumulator_all);
                HWY_AFTER_TEST();
            }
        }
    }
}
HWY_TEST_MAIN();

#endif