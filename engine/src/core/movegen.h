#ifndef CHEPP_MOVEGEN_H
#define CHEPP_MOVEGEN_H

#include "bitboard.h"

#include "generated/from_to.h"
#include "generated/lines.h"
#if USE_PEXT
#include "generated/magic_bishops_pext.h"
#include "generated/magic_rooks_pext.h"
#else
#include "generated/magic_bishops.h"
#include "generated/magic_rooks.h"
#endif

#include <bit>

namespace chepp::movegen {
    namespace detail {
        inline Magics<BISHOP> G_MAGIC_BISHOP{};
        inline Magics<ROOK> G_MAGIC_ROOK{};
        inline EnumArray<Bitboard, Square, Square> LINES{};
        inline EnumArray<Bitboard, Square, Square> FROM_TO{};
    } // namespace detail

    inline static void init() {
        static std::once_flag init_once;
        std::call_once(init_once, [] {
#if USE_PEXT
            detail::G_MAGIC_BISHOP.read(GENERATED_MAGIC_BISHOPS_PEXT.begin());
            detail::G_MAGIC_ROOK.read(GENERATED_MAGIC_ROOKS_PEXT.begin());
#else
            detail::G_MAGIC_BISHOP.read(GENERATED_MAGIC_BISHOPS.begin());
            detail::G_MAGIC_ROOK.read(GENERATED_MAGIC_ROOKS.begin());
#endif
            utils::read_range(detail::LINES, GENERATED_LINES.begin());
            utils::read_range(detail::FROM_TO, GENERATED_FROM_TO.begin());
        });
    }

    struct Initializer {
        Initializer() { init(); }
    };

    inline Initializer g_initializer;

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

    inline static const EnumArray<Bitboard (*)(Square, Bitboard, Color), PieceType> attack_table{
        constexpr_in_place, []<PieceType pt>(std::integral_constant<PieceType, pt>) { return &attacks<pt>; }};

    inline constexpr Bitboard
    attacks(const PieceType pt, const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE) {
        return attack_table.at(pt)(sq, occupancy, c);
    }
} // namespace chepp::movegen

#endif // CHEPP_MOVEGEN_H
