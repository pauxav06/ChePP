#ifndef CHEPP_MOVEGEN_H
#define CHEPP_MOVEGEN_H

#include "bitboard.h"

#include "generated/from_to.h"
#include "generated/lines.h"
#include "generated/magic_bishops.h"
#include "generated/magic_rooks.h"

namespace chepp::movegen {
    namespace detail {
        inline constexpr auto G_MAGIC_BISHOP = std::bit_cast<Magics<BISHOP>>(GENERATED_MAGIC_BISHOPS);
        inline constexpr auto G_MAGIC_ROOK   = std::bit_cast<Magics<ROOK>>(GENERATED_MAGIC_ROOKS);
        inline constexpr auto LINES          = std::bit_cast<EnumArray<Bitboard, Square, Square>>(GENERATED_LINES);
        inline constexpr auto FROM_TO        = std::bit_cast<EnumArray<Bitboard, Square, Square>>(GENERATED_FROM_TO);
    } // namespace detail

    inline constexpr Bitboard
    line(const Square sq1, const Square sq2) {
        return detail::LINES[sq1][sq2];
    }

    inline constexpr bool
    are_aligned(const Square sq1, const Square sq2, const Square sq3) {
        return line(sq1, sq2) == line(sq2, sq3);
    }

    inline constexpr Bitboard
    from_to_incl(const Square sq1, const Square sq2) {
        return detail::FROM_TO[sq1][sq2];
    }

    inline constexpr Bitboard
    from_to_excl(const Square sq1, const Square sq2) {
        return from_to_incl(sq1, sq2).unset(sq1).unset(sq2);
    }

    template <PieceType pc>
        requires(pc != NO_PIECE_TYPE)
    inline constexpr Bitboard
    pseudo_attack(const Square sq, const Color c) noexcept {
        if constexpr (pc == PAWN) {
            return detail::PAWN_PSEUDO_ATTACKS[c][sq];
        } else if constexpr (pc == KNIGHT || pc == BISHOP || pc == ROOK || pc == QUEEN || pc == KING) {
            return detail::PIECE_PSEUDO_ATTACKS[pc][sq];
        }
        return {};
    }

    template <PieceType pc>
        requires(pc != NO_PIECE_TYPE)
    inline constexpr Bitboard
    attacks(const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE) noexcept {
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

    inline static constinit EnumArray<Bitboard (*)(Square, Bitboard, Color), PieceType> attack_table{
        constexpr_in_place, []<PieceType pt>(std::integral_constant<PieceType, pt>) { return &attacks<pt>; }};

    inline constexpr Bitboard
    attacks(const PieceType pt, const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE) {
        return attack_table.at(pt)(sq, occupancy, c);
    }
} // namespace chepp::movegen

#endif // CHEPP_MOVEGEN_H
