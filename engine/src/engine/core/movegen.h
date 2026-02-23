#ifndef CHEPP_MOVEGEN_H
#define CHEPP_MOVEGEN_H

#include "bitboard.h"

#include "magics.h"

namespace chepp::movegen {
    namespace detail {
        template <PieceType pc>
        const Magics<pc>&
        magics();

        template <>
        inline const Magics<BISHOP>&
        magics<BISHOP>() {
            return G_MAGIC_BISHOP;
        }

        template <>
        inline const Magics<ROOK>&
        magics<ROOK>() {
            return G_MAGIC_ROOK;
        }
    } // namespace detail

    template <PieceType pc>
    constexpr Bitboard
    attacks(const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE) {
        static_assert(pc, "Invalid piece type");
        if constexpr (pc == ROOK || pc == BISHOP) {
            return detail::magics<pc>().attack(sq, occupancy);
        } else if constexpr (pc == QUEEN) {
            return attacks<BISHOP>(sq, occupancy) | attacks<ROOK>(sq, occupancy);
        } else if constexpr (pc == PAWN || pc == KNIGHT || pc == KING) {
            return pseudo_attack<pc>(sq, c);
        }
        return {};
    }

    inline Bitboard
    attacks(const PieceType pt, const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE) {
        switch (pt.value()) {
            case (PAWN).value():
                return attacks<PAWN>(sq, occupancy, c);
            case (KNIGHT).value():
                return attacks<KNIGHT>(sq, occupancy, c);
            case (BISHOP).value():
                return attacks<BISHOP>(sq, occupancy, c);
            case (ROOK).value():
                return attacks<ROOK>(sq, occupancy, c);
            case (QUEEN).value():
                return attacks<QUEEN>(sq, occupancy, c);
            case (KING).value():
                return attacks<KING>(sq, occupancy, c);
            default:
                assert(false && "invalid piece type");
                return bb::empty();
        }
    }
} // namespace chepp::movegen

#endif // CHEPP_MOVEGEN_H
