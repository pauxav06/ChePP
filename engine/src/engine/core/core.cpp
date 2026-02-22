#include "bitboard.h"
#include <bit>
#include <cstdint>

extern "C" uint32_t
tb_cpp_popcnt32(uint32_t x) {
    return std::popcount(x);
}

extern "C" uint64_t
tb_cpp_popcnt64(uint64_t x) {
    return std::popcount(x);
}

extern "C" uint32_t
tb_bswap32(uint32_t x) {
    return std::byteswap(x);
}

extern "C" uint64_t
tb_bswap64(uint64_t x) {
    return std::byteswap(x);
}

extern "C" uint32_t
tb_lsb32(uint32_t x) {
    return std::countr_zero(x);
}

extern "C" uint64_t
tb_king_attacks(int x) {
    return Movegen::attacks<KING>(Square{x}).value();
}
