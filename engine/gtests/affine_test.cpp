#include <random>
#include <stdint.h>
#include "tune.h"

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
                void fill_random(hwy::AlignedVector<T>& v, int min, int max, unsigned seed = 1234)
                {
                    std::mt19937                       rng(seed);
                    std::uniform_int_distribution<int> dist(min, max);
                    for (auto& x : v)
                        x = static_cast<T>(dist(rng));
                }

                void reference_affine(const hwy::AlignedVector<int8_t>& input, const hwy::AlignedVector<int8_t>& weights,
                                      const hwy::AlignedVector<int32_t>& biases, size_t Rows, size_t Cols,
                                      hwy::AlignedVector<int32_t>& out)
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
                using namespace hn;
                using namespace meta;

                void affine()
                {
                    std::cout << "Test passed for target: " << hwy::TargetName(HWY_TARGET) << std::endl;
                    constexpr size_t in       = 2048;
                    constexpr size_t out       = 16;
                    constexpr size_t Iterations = 10000;

                    hwy::AlignedVector<int8_t>  weights(in * out);
                    hwy::AlignedVector<int32_t> biases(out);
                    hwy::AlignedVector<int8_t>  input(in);
                    hwy::AlignedVector<int32_t> ref_output(out);
                    hwy::AlignedVector<int32_t> simd_output(out);

                    fill_random(weights, -3, 3);
                    fill_random(biases, -10, 10);
                    fill_random(input, 0, 127);

                    using namespace std::chrono;
                    int64_t total_ref_duration = 0;
                    auto start_ref = high_resolution_clock::now();
                    for (size_t iter = 0; iter < Iterations; ++iter)
                    {
                        reference_affine(input, weights, biases, out, in, ref_output);
                    }
                    auto end_ref = high_resolution_clock::now();
                    total_ref_duration += duration_cast<nanoseconds>(end_ref - start_ref).count();

                    constexpr Types types = {
                        .in = ScalarType::Int8,
                        .out = ScalarType::Int32,
                    };

                    constexpr Dims dims = {
                        .in = in,
                        .out = out,
                    };

                    using opt_t   = Opt<Kernels::SIMD>;
                    using param_t = Params<Kernels::SIMD>;
                    constexpr std::array type_arr   = {types};
                    constexpr std::array dim_arr    = {dims};
                    constexpr std::array unroll     = {1, 2, 4, 8, 16};
                    constexpr std::array operations = {opt_t::Operation::SumOfMulQuadAcc,
                                                       opt_t::Operation::SumOfMulPairAdd};

                    constexpr auto configs =
                        tune::generate_combinations<param_t>(
                            type_arr, dim_arr,
                            tune::generate_combinations<opt_t>(unroll, operations));

                    auto run_all_configs = [&](auto&& input, auto&& weights, auto&& biases,
                                               auto&& simd_output, auto&& ref_output) {
                        [&]<size_t... I>(std::index_sequence<I...>) {
                            (([&]{
                                constexpr auto config = configs[I];
                                using layer_t = Layer<Kernels::SIMD, config>;
                                layer_t layer;
                                layer.load_weights(weights.data(), biases.data());

                                int64_t total_simd_duration = 0;
                                auto start_simd = high_resolution_clock::now();
                                for (size_t iter = 0; iter < Iterations; ++iter)
                                    layer.forward(input.data(), simd_output.data());
                                auto end_simd = high_resolution_clock::now();
                                total_simd_duration += duration_cast<nanoseconds>(end_simd - start_simd).count();

                                reference_affine(input, weights, biases, out, in, ref_output);
                                HWY_ASSERT_ARRAY_EQ(simd_output.data(), ref_output.data(), simd_output.size());

                                std::cout << "Reference (naive) affine avg: " << total_ref_duration / Iterations << " ns\n";
                                std::cout << "SIMD affine layer avg:        " << total_simd_duration / Iterations << " ns\n";
                            }()), ...);
                        }(std::make_index_sequence<configs.size()>{});
                    };

                    run_all_configs(input, weights, biases, simd_output, ref_output);



                }

                void affine_all()
                {
                    affine();
                }
            }
        }
    }

    /**
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
    **/
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
    /**
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
    **/
}
HWY_TEST_MAIN();
#endif // HWY_ONCE