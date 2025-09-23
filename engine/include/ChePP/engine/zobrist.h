#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "types.h"

#include <cstdint>
#include <array>
#include <random>
#include <mutex>

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

    inline const auto& psq_table()
    {
        static const auto& table = [] {
            EnumArray<Hash, Piece, Square> t{};
            auto& gen = prng::thread_local_gen();
            t.fill_pred([&gen]([[maybe_unused]] auto _, [[maybe_unused]] auto __) {
                return prng::next_u64(gen);
            });
            return t;
        }();
        return table;
    }

    inline const auto& ep_table()
    {
        static const auto& table = [] {
            EnumArray<Hash, File> t{};
            auto& gen = prng::thread_local_gen();
            t.fill_pred([&gen]([[maybe_unused]] auto _) {
                return prng::next_u64(gen);
            });
            return t;
        }();
        return table;
    }

    inline const auto& castling_table()
    {
        static const auto& table = [] {
            EnumArray<Hash, CastlingType> t{};
            auto& gen = prng::thread_local_gen();
            t.fill_pred([&gen]([[maybe_unused]] auto _) {
                return prng::next_u64(gen);
            });
            return t;
        }();
        return table;
    }

    inline Hash side_table()
    {
        static const Hash value = [] {
            auto& gen = prng::thread_local_gen();
            return prng::next_u64(gen);
        }();
        return value;
    }

    inline Hash no_pawns_table()
    {
        static const Hash value = [] {
            auto& gen = prng::thread_local_gen();
            return prng::next_u64(gen);
        }();
        return value;
    }

    inline void flip_piece(Hash& hash, const Piece pt, const Square sq)
    {
        hash ^= psq_table().at(pt).at(sq);
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
                hash ^= castling_table().at(type);
        }
    }

    inline void flip_ep(Hash& hash, const File fl)
    {
        hash ^= ep_table().at(fl);
    }

    inline void flip_color(Hash& hash)
    {
        hash ^= side_table();
    }

} // namespace zobrist

#endif // ZOBRIST_H
