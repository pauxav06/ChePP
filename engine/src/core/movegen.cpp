#include "movegen.h"

#if USE_PEXT
#include "generated/magic_bishops_pext.h"
#include "generated/magic_rooks_pext.h"
#else
#include "generated/magic_bishops.h"
#include "generated/magic_rooks.h"
#endif

namespace chepp::movegen::detail {
    template <PieceType pc>
    using Magics = MagicsBase<pc, std::conditional_t<USE_PEXT, PEXTIndexer, ShiftIndexer>>;

    static constinit Magics<ROOK> G_MAGIC_ROOK{};
    static constinit Magics<BISHOP> G_MAGIC_BISHOP{};

    void init_magics() noexcept {
#if USE_PEXT
        G_MAGIC_BISHOP.read(GENERATED_MAGIC_BISHOPS_PEXT.begin());
        G_MAGIC_ROOK.read(GENERATED_MAGIC_ROOKS_PEXT.begin());
#else
        G_MAGIC_BISHOP.read(GENERATED_MAGIC_BISHOPS.begin());
        G_MAGIC_ROOK.read(GENERATED_MAGIC_ROOKS.begin());
#endif
    }

    Bitboard rook_attacks(Square sq, Bitboard occ) noexcept {
        return G_MAGIC_ROOK.attack(sq, occ);
    }

    Bitboard bishop_attacks(Square sq, Bitboard occ) noexcept {
        return G_MAGIC_BISHOP.attack(sq, occ);
    }
}
