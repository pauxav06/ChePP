#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "types.h"

#include <cstdint>
#include <functional>

struct ConstexprPRNG {
    using result_type = std::uint64_t;

    std::uint64_t state;

    constexpr explicit ConstexprPRNG(std::uint64_t seed) : state(seed) {
    }

    constexpr std::uint64_t
    next() {
        std::uint64_t z = (state += 0x9E3779B97f4A7C15ull);
        z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    constexpr std::uint64_t
    next(std::uint64_t max) {
        return next() % max;
    }
};

namespace chepp::zobrist {
    using Hash = uint64_t;

    template <std::size_t seed, typename... TS>
    struct RandomHashTable {
        inline static constexpr EnumArray<Hash, TS...> table {
            std::in_place, [rng = ConstexprPRNG{seed}] (auto...) mutable {
                return rng.next();
            }
        };
    };

    using PSQ = RandomHashTable<0x1234, Piece, Square>;
    using EP = RandomHashTable<0x432, File>;
    using CASTLING = RandomHashTable<0x888, CastlingType>;
    using SIDE = RandomHashTable<0x111111, Color>;

    inline constexpr void
    flip_piece(Hash& hash, const Piece pt, const Square sq) {
        hash ^= PSQ::table.at(pt).at(sq);
    }

    inline constexpr void
    move_piece(Hash& hash, const Piece pt, const Square from, const Square to) {
        flip_piece(hash, pt, from);
        flip_piece(hash, pt, to);
    }

    inline constexpr void
    promote_piece(Hash& hash, const Color c, const PieceType pt, const Square sq) {
        flip_piece(hash, Piece{c, PAWN}, sq);
        flip_piece(hash, Piece{c, pt}, sq);
    }

    inline constexpr void
    flip_castling_rights(Hash& hash, const uint8_t mask) {
        for (auto type : {WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE}) {
            if (mask & type.mask()) hash ^= CASTLING::table.at(type);
        }
    }

    inline constexpr void
    flip_ep(Hash& hash, const File fl) {
        hash ^= EP::table.at(fl);
    }

    inline constexpr void
    flip_color(Hash& hash) {
        hash ^= SIDE::table.at(WHITE);
    }

} // namespace chepp::zobrist

#endif // ZOBRIST_H
