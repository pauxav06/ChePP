#include "bitboard.h"

#if CHEPP_PEXT
#include <immintrin.h>
#endif

namespace chepp {
    uint64_t
    pext(const uint64_t val, const uint64_t mask) noexcept {
#if CHEPP_PEXT
        return _pext_u64(val, mask);
#else
        (void)val;
        (void)mask;
        fmt::println(std::cerr, "Unsuported pext");
        std::terminate();
#endif
    }
} // namespace chepp