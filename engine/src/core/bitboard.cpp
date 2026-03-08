#include "bitboard.h"
#include "format.h"

#if CHEPP_PEXT
#include <immintrin.h>
#endif

namespace chepp {
#if CHEPP_PEXT
    uint64_t
    pext(const uint64_t val, const uint64_t mask) noexcept {
        return _pext_u64(val, mask);
    }
#endif
} // namespace chepp