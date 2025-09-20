#pragma once
#include <cstdint>

alignas(64) extern int16_t g_ft_weights[23068672];
alignas(64) extern int16_t g_ft_biases[1024];
alignas(64) extern int16_t g_psqt_weights[180224];
alignas(64) extern int16_t g_psqt_biases[8];
alignas(64) extern int16_t g_l1_psqt_weights[16384];
alignas(64) extern int32_t g_l1_psqt_biases[8];
alignas(64) extern int16_t g_l1_weights[262144];
alignas(64) extern int32_t g_l1_biases[128];
alignas(64) extern int16_t g_l2_weights[4096];
alignas(64) extern int32_t g_l2_biases[256];
alignas(64) extern int16_t g_out_weights[256];
alignas(64) extern int32_t g_out_bias[8];
