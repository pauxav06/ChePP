#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "types.h"

#include <cstdint>
#include <random>

namespace prng
{
    inline std::mt19937_64& thread_local_gen()
    {
        thread_local std::mt19937_64 gen{std::random_device{}()};
        return gen;
    }

    inline uint64_t next_u64(std::mt19937_64& gen)
    {
        return gen();
    }
}

namespace Zobrist
{
    using Hash = uint64_t;

    inline EnumArray<Hash, Piece, Square> PSQ_TABLE;
    inline EnumArray<Hash, File> EP_TABLE;
    inline EnumArray<Hash, CastlingType> CASTLING_TABLE;
    inline Hash SIDE_TABLE = 0;

    struct Initialiser
    {
        Initialiser()
        {
            auto& gen = prng::thread_local_gen();

            PSQ_TABLE.fill_pred([&gen]([[maybe_unused]] auto, [[maybe_unused]] auto) {
                return prng::next_u64(gen);
            });

            EP_TABLE.fill_pred([&gen]([[maybe_unused]] auto) {
                return prng::next_u64(gen);
            });

            CASTLING_TABLE.fill_pred([&gen]([[maybe_unused]] auto) {
                return prng::next_u64(gen);
            });

            SIDE_TABLE = prng::next_u64(gen);
        }
    };

    inline Initialiser _zobrist_initializer;

    inline void flip_piece(Hash& hash, const Piece pt, const Square sq)
    {
        hash ^= PSQ_TABLE.at(pt).at(sq);
    }

    inline void move_piece(Hash& hash, const Piece pt, const Square from, const Square to)
    {
        flip_piece(hash, pt, from);
        flip_piece(hash, pt, to);
    }

    inline void promote_piece(Hash& hash, const Color c, const PieceType pt, const Square sq)
    {
        flip_piece(hash, Piece{c, PAWN}, sq);
        flip_piece(hash, Piece{c, pt}, sq);
    }

    inline void flip_castling_rights(Hash& hash, const uint8_t mask)
    {
        for (auto type : {WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE})
        {
            if (mask & type.mask())
                hash ^= CASTLING_TABLE.at(type);
        }
    }

    inline void flip_ep(Hash& hash, const File fl)
    {
        hash ^= EP_TABLE.at(fl);
    }

    inline void flip_color(Hash& hash)
    {
        hash ^= SIDE_TABLE;
    }

    inline Hash side_table()
    {
        return SIDE_TABLE;
    }

} // namespace Zobrist

#endif // ZOBRIST_H
