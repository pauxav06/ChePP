#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "prng.h"
#include "types.h"

#include <charconv>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <ostream>

using hash_t = uint64_t;

struct zobrist_t
{
    explicit zobrist_t(const hash_t& hash) : m_hash(hash) {}
    zobrist_t(const zobrist_t& other) = default;
    explicit zobrist_t()              = default;

    static void init(std::uint64_t seed);

    [[nodiscard]] hash_t value() const { return m_hash; }

    void flip_piece(const Piece pt, const Square sq) { m_hash ^= s_psq.at(pt).at(sq); }

    void move_piece(const Piece pt, const Square from, const Square to)
    {
        flip_piece(pt, from);
        flip_piece(pt, to);
    }

    void promote_piece(const Color c, const PieceType pt, const Square sq)
    {
        flip_piece(Piece{c, PAWN}, sq);
        flip_piece(Piece{c, pt}, sq);
    }

    void flip_castling_rights(const uint8_t mask)
    {
        for (auto type : {WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE})
        {
            if (mask & type.mask())
                m_hash ^= s_castling.at(type);
        }
    }

    void flip_ep(const File fl) { m_hash ^= s_ep.at(fl); }

    void flip_color() { m_hash ^= s_side; }

    static const EnumArray<Piece, EnumArray<Square, hash_t>> s_psq;
    static const EnumArray<File, hash_t>                     s_ep;
    static const EnumArray<CastlingType, hash_t>             s_castling;
    static const hash_t                                      s_side, s_no_pawns;

    hash_t m_hash{};
};

constexpr PRNG master_gen{make_seed_(__FILE__, "", __LINE__)};

constexpr auto init_zobrist_tables()
{
    PRNG gen = master_gen;

    EnumArray<Piece, EnumArray<Square, hash_t>> psq{};
    for (const Piece pc : Piece::values())
    {
        for (const Square sq : Square::values())
        {
            gen = gen.next(psq.at(pc).at(sq));
        }
    }

    EnumArray<File, hash_t> ep{};
    for (const File fl : File::values())
    {
        gen = gen.next(ep.at(fl));
    }

    EnumArray<CastlingType, hash_t> castling{};
    for (const CastlingType ct : CastlingType::values())
    {
        gen = gen.next(castling.at(ct));
    }

    uint64_t side{};
    gen = gen.next(side);

    uint64_t no_pawns{};
    gen.next(no_pawns);

    return std::make_tuple(psq, ep, castling, side, no_pawns);
}

constexpr auto zobrist_tables = init_zobrist_tables();

constexpr EnumArray<Piece, EnumArray<Square, hash_t>> zobrist_t::s_psq      = std::get<0>(zobrist_tables);
constexpr EnumArray<File, hash_t>                     zobrist_t::s_ep       = std::get<1>(zobrist_tables);
constexpr EnumArray<CastlingType, hash_t>             zobrist_t::s_castling = std::get<2>(zobrist_tables);
constexpr hash_t                                      zobrist_t::s_side     = std::get<3>(zobrist_tables);
constexpr hash_t                                      zobrist_t::s_no_pawns = std::get<4>(zobrist_tables);

#endif // ZOBRIST_H
