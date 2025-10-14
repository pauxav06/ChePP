#pragma once
#include <cstddef>
#include <cstdint>

alignas(64) extern const int16_t acc_weights[1][23068672];
alignas(64) extern const int16_t acc_biases[1][1024];
alignas(64) extern const int32_t psqt_weights[1][180224];
alignas(64) extern const int32_t psqt_biases[1][8];
alignas(64) extern const int8_t l1_weights[8][32768];
alignas(64) extern const int32_t l1_biases[8][16];
alignas(64) extern const int8_t l2_weights[8][512];
alignas(64) extern const int32_t l2_biases[8][32];
alignas(64) extern const int8_t out_weights[8][32];
alignas(64) extern const int32_t out_biases[8][1];
