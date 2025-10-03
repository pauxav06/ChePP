//
// Created by paul on 10/2/25.
//
#include <ChePP/nnue/layers.h>
#include <gtest/gtest.h>

#include <random>
#include <algorithm>
#include <ranges>

template <typename T>
void fill_random(std::vector<T>& v, int min, int max, unsigned seed = 1234) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(min, max);
    for (auto& x : v) x = static_cast<T>(dist(rng));
}

void reference_affine(const std::vector<int8_t>& input,
                      const std::vector<int8_t>& weights,
                      const std::vector<int32_t>& biases,
                      size_t Rows, size_t Cols,
                      std::vector<int32_t>& out) {
    out.assign(Rows, 0);
    for (size_t r = 0; r < Rows; r++) {
        int32_t acc = biases[r];
        for (size_t c = 0; c < Cols; c++) {
            acc += static_cast<int32_t>(input[c]) * static_cast<int32_t>(weights[r * Cols + c]);
        }
        out[r] = acc;
    }
}

template <typename T>
int8_t clip_relu_ref(T x) {
    if (x < 0) return 0;
    if (x > 127) return 127;
    return static_cast<int8_t>(x);
}
TEST(AffineLayerTest, SmallDenseLayer) {
    constexpr size_t Rows = 16;
    constexpr size_t Cols = 2048;
    constexpr size_t Iterations = 10000;

    std::vector<int8_t> weights(Rows * Cols);
    std::vector<int32_t> biases(Rows);
    std::vector<int8_t> input(Cols);
    std::vector<int32_t> ref_output(Rows);
    std::vector<int32_t> simd_output(Rows);

    fill_random(weights, -3, 3);
    fill_random(biases, -10, 10);
    fill_random(input, 0, 127);

    using namespace std::chrono;
    int64_t total_ref_duration = 0;
    for (size_t iter = 0; iter < Iterations; ++iter) {
        auto start_ref = high_resolution_clock::now();
        reference_affine(input, weights, biases, Rows, Cols, ref_output);
        auto end_ref = high_resolution_clock::now();
        total_ref_duration += duration_cast<nanoseconds>(end_ref - start_ref).count();
    }

    using layer_t = affine::AffineLayer<Rows, Cols>;
    layer_t layer;
    layer.load_weights(weights.data(), biases.data());

    int64_t total_simd_duration = 0;
    for (size_t iter = 0; iter < Iterations; ++iter) {
        auto start_simd = high_resolution_clock::now();
        layer.forward(input.data(), simd_output.data());
        auto end_simd = high_resolution_clock::now();
        total_simd_duration += duration_cast<nanoseconds>(end_simd - start_simd).count();
    }

    reference_affine(input, weights, biases, Rows, Cols, ref_output);
    for (size_t i = 0; i < Rows; i++) {
        EXPECT_EQ(simd_output[i], ref_output[i]) << "Mismatch at row " << i;
    }

    std::cout << "Reference (naive) affine avg: " << total_ref_duration / Iterations << " ns\n";
    std::cout << "SIMD affine layer avg:        " << total_simd_duration / Iterations << " ns\n";
}


TEST(ClippedRelu16_8, PositiveAndNegative) {
    constexpr size_t N = 16;
    std::vector<int16_t> input(N);
    std::vector<int8_t> output(N);
    std::vector<int8_t> ref(N);

    input = {-200, -1, 0, 1, 127, 128, 30000, -32768,
             255, 500, 1000, -999, 42, 99, -100, 32767};

    relu::QuantizedClippedRelu16_8<N>::forward(input.data(), output.data());

    for (size_t i = 0; i < N; i++) {
        ref[i] = clip_relu_ref(input[i]);
        EXPECT_EQ(output[i], ref[i]) << "Mismatch at index " << i;
    }
}

TEST(ClippedRelu32_8, PositiveAndNegative) {
    constexpr size_t N = 32;
    std::vector<int32_t> input(N);
    std::vector<int8_t> output(N);
    std::vector<int8_t> ref(N);

    input = {-200, -1, 0, 1, 127, 128, 255, 1000,
             32767, -32768, 999999, -999999, 42, 99, -100, 2147483647,
             -2147483648, 50, 75, 300, 60000, -40000, 8192, -8192,
             7, -7, 13, -13, 5000, -5000, 100, -100};

    relu::ClippedRelu32_8<N>::forward(input.data(), output.data());

    for (size_t i = 0; i < N; i++) {
        ref[i] = clip_relu_ref(input[i]);
        EXPECT_EQ(output[i], ref[i]) << "Mismatch at index " << i;
    }
}

TEST(ClippedRelu16_8, BigRandom) {
    constexpr size_t N = 1024;
    std::vector<int16_t> input(N);
    std::vector<int8_t> output(N);
    std::vector<int8_t> ref(N);

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int16_t> dist(std::numeric_limits<int16_t>::min(),
                                                 std::numeric_limits<int16_t>::max());
    for (auto &x : input) x = dist(rng);

    relu::QuantizedClippedRelu16_8<N>::forward(input.data(), output.data());

    for (size_t i = 0; i < N; i++) {
        ref[i] = clip_relu_ref(input[i]);
        EXPECT_EQ(output[i], ref[i]) << "Mismatch at index " << i
                                     << " input=" << input[i];
    }
}

TEST(ClippedRelu32_8, BigRandom) {
    constexpr size_t N = 1024;
    std::vector<int32_t> input(N);
    std::vector<int8_t> output(N);
    std::vector<int8_t> ref(N);

    std::mt19937 rng(5678);
    std::uniform_int_distribution<int32_t> dist(std::numeric_limits<int32_t>::min(),
                                                 std::numeric_limits<int32_t>::max());
    for (auto &x : input) x = dist(rng);

    relu::ClippedRelu32_8<N>::forward(input.data(), output.data());

    for (size_t i = 0; i < N; i++) {
        ref[i] = clip_relu_ref(input[i]);
        EXPECT_EQ(output[i], ref[i]) << "Mismatch at index " << i
                                     << " input=" << input[i];
    }
}

