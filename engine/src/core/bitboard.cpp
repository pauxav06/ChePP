#include "bitboard.h"
#include "format.h"

#if CHEPP_PEXT
#include <immintrin.h>
#endif

namespace chepp {
    uint64_t
    pext(const uint64_t val, uint64_t mask) noexcept {
#if USE_PEXT
        return _pext_u64(val, mask);
#else
        uint64_t res = 0;
        uint64_t bb  = 1;

        while (mask)
        {
            uint64_t lowest = mask & -mask;

            if (val & lowest)
                res |= bb;

            mask ^= lowest;
            bb <<= 1;
        }

        return res;
#endif
    }
} // namespace chepp