#pragma once
#include <cstdint>

alignas(64) extern int16_t g_ft_weights[12582912];
alignas(64) extern int16_t g_ft_biases[1024];
alignas(64) extern int16_t g_l1_weights[32768];
alignas(64) extern int32_t g_l1_biases[16];
alignas(64) extern int16_t g_l2_weights[512];
alignas(64) extern int32_t g_l2_biases[32];
alignas(64) extern int16_t g_out_weights[32];
alignas(64) extern int32_t g_out_bias[1];
