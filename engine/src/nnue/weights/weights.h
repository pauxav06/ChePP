#pragma once
#include <array>
#include <cstdint>

#if defined(IDE) && IDE == 1
inline constinit const std::array<uint8_t, 47128384> GENERATED_WEIGHTS{};
#else
#include "/home/paul/code/ChePP/engine/src/nnue/weights/weights.inc"
#endif
