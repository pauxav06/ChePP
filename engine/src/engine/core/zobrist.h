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

template <std::size_t N, uint64_t seed>
constexpr auto
generate() {
    ConstexprPRNG           rng(seed);
    std::array<uint64_t, N> arr{};
    for (std::size_t i = 0; i < N; i++) {
        arr[i] = rng.next();
    }
    return arr;
}

namespace chepp::zobrist {
    using Hash = uint64_t;

    template <std::size_t seed, typename... TS>
    struct RandomHahsTable {
        static constexpr EnumArray<Hash, TS...> table{
            std::in_place, [values = generate<EnumArray<Hash, TS...>::flat_size(), seed>()](auto... idx) {
                return values.at(EnumArray<Hash, TS...>::flat_index(idx...));
            }};
    };

    using PSQ_TABLE      = RandomHahsTable<__LINE__, Piece, Square>;
    using EP_TABLE       = RandomHahsTable<__LINE__, File>;
    using CASTLING_TABLE = RandomHahsTable<__LINE__, CastlingType>;
    using SIDE_TABLE     = RandomHahsTable<__LINE__, Color>;

    inline constexpr void
    flip_piece(Hash& hash, const Piece pt, const Square sq) {
        hash ^= PSQ_TABLE::table.at(pt).at(sq);
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
            if (mask & type.mask()) hash ^= CASTLING_TABLE::table.at(type);
        }
    }

    inline constexpr void
    flip_ep(Hash& hash, const File fl) {
        hash ^= EP_TABLE::table.at(fl);
    }

    inline constexpr void
    flip_color(Hash& hash) {
        hash ^= SIDE_TABLE::table.at(WHITE);
    }

} // namespace chepp::zobrist

#endif // ZOBRIST_H
