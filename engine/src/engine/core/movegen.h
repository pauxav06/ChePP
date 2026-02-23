#ifndef CHEPP_MOVEGEN_H
#define CHEPP_MOVEGEN_H

#include "bitboard.h"
#include "generated/bishop.h"
#include "generated/from_to.h"
#include "generated/lines.h"
#include "generated/rook.h"

namespace chepp::movegen {
    namespace detail {
        inline constexpr auto G_MAGIC_BISHOP = std::bit_cast<Magics<BISHOP>>(GENERATED_BISHOP);
        inline constexpr auto G_MAGIC_ROOK   = std::bit_cast<Magics<ROOK>>(GENERATED_ROOK);
        static constexpr auto LINES          = std::bit_cast<EnumArray<Bitboard, Square, Square>>(GENERATED_LINES);
        static constexpr auto FROM_TO        = std::bit_cast<EnumArray<Bitboard, Square, Square>>(GENERATED_FROM_TO);
    } // namespace detail

    static constexpr Bitboard
    line(const Square sq1, const Square sq2) {
        return detail::LINES[sq1][sq2];
    }

    static constexpr bool
    are_aligned(const Square sq1, const Square sq2, const Square sq3) {
        return line(sq1, sq2) == line(sq2, sq3);
    }

    static constexpr Bitboard
    from_to_incl(const Square sq1, const Square sq2) {
        return detail::FROM_TO[sq1][sq2];
    }
    static constexpr Bitboard
    from_to_excl(const Square sq1, const Square sq2) {
        return from_to_incl(sq1, sq2).unset(sq1).unset(sq2);
    }

    template <PieceType pc>
    constexpr Bitboard
    pseudo_attack(const Square sq, const Color c) {
        static_assert(pc != NO_PIECE_TYPE, "Invalid piece type");
        if constexpr (pc == PAWN)
            return detail::PAWN_PSEUDO_ATTACKS[c][sq];
        else if constexpr (pc == KNIGHT || pc == BISHOP || pc == ROOK || pc == QUEEN || pc == KING)
            return detail::PIECE_PSEUDO_ATTACKS[pc][sq];
        return bb::empty();
    }

    template <PieceType pc>
    constexpr Bitboard
    attacks(const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE) {
        static_assert(pc, "Invalid piece type");
        if constexpr (pc == BISHOP) {
            return detail::G_MAGIC_BISHOP.attack(sq, occupancy);
        } else if constexpr (pc == ROOK) {
            return detail::G_MAGIC_ROOK.attack(sq, occupancy);
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
