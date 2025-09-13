#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include "prng.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <format>
#include <unordered_set>
#include <vector>

struct Position;

// a custom enum of consecutive integers 0 ... N and a NONE value N + 1
template <typename DerivedT, typename UnderlyingT, std::size_t COUNT,
          const std::array<std::string_view, COUNT + 1>& repr, bool EnableInc = false, bool EnableArithmetic = false>
    requires std::is_integral_v<UnderlyingT>
struct EnumBase
{
    static constexpr bool EnableInc_v        = EnableInc;
    static constexpr bool EnableArithmetic_v = EnableArithmetic;

    static constexpr std::size_t COUNT_V = COUNT;
    static constexpr std::size_t TOTAL_V = COUNT + 1;
    static constexpr const auto& repr_v  = repr;

    using ValueT = UnderlyingT;
    using ReprT  = const std::array<std::string_view, TOTAL_V>; // array of string representation of values
    using IndexT = std::size_t;

    static constexpr auto& repr_f() { return DerivedT::repr_v; }

    // we do not use 0 for NONE because packing is easier if useful values occupy lower bits
    static constexpr ValueT NONE_VALUE = static_cast<ValueT>(COUNT);
    explicit constexpr      operator bool() const noexcept { return m_val != NONE_VALUE; }

    constexpr EnumBase() : m_val(NONE_VALUE) {};

    template <typename int_type, std::enable_if_t<std::is_integral_v<int_type>, int> = 0>
    constexpr explicit EnumBase(int_type v) : m_val(std::min(static_cast<ValueT>(v), NONE_VALUE))
    {
    }

    [[nodiscard]] constexpr ValueT             value() const noexcept { return m_val; }
    [[nodiscard]] constexpr IndexT             index() const noexcept { return static_cast<IndexT>(m_val); }
    [[nodiscard]] constexpr bool               is_none() const noexcept { return m_val == NONE_VALUE; }
    [[nodiscard]] static constexpr std::size_t count() noexcept { return COUNT_V; }
    [[nodiscard]] static constexpr std::size_t total() noexcept { return TOTAL_V; }

    [[nodiscard]] constexpr std::string_view to_string() const { return repr.at(index()); }

    [[nodiscard]] static constexpr std::optional<DerivedT> from_string(const std::string_view& sv)
    {
        const auto it = std::ranges::find(repr, sv);
        return it == repr.end() ? std::nullopt : std::optional{DerivedT(std::distance(repr.begin(), it))};
    }

    friend std::ostream& operator<<(std::ostream& os, const EnumBase& e)
    {
        os << e.to_string();
        return os;
    }

    static constexpr std::array<DerivedT, COUNT_V> values()
    {
        std::array<DerivedT, COUNT_V> arr{};
        for (std::size_t i = 0; i < COUNT_V; ++i)
            arr[i] = DerivedT(static_cast<ValueT>(i));
        return arr;
    };

    friend constexpr DerivedT operator+(const DerivedT a, const int rhs) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) + rhs));
    }

    friend constexpr DerivedT operator-(const DerivedT a, const int rhs) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) - rhs));
    }

    friend constexpr DerivedT operator+(const DerivedT a, const DerivedT b) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) + static_cast<int>(b.m_val)));
    }

    friend constexpr DerivedT operator-(const DerivedT a, const DerivedT b) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) - static_cast<int>(b.m_val)));
    }

    friend constexpr DerivedT& operator++(DerivedT& d) noexcept
        requires(EnableInc)
    {
        d.m_val = static_cast<ValueT>(d.m_val + 1);
        return d;
    }

    friend constexpr DerivedT& operator--(DerivedT& d) noexcept
        requires(EnableInc)
    {
        d.m_val = static_cast<ValueT>(d.m_val - 1);
        return d;
    }

    friend constexpr DerivedT operator++(DerivedT& d, int) noexcept
        requires(EnableInc)
    {
        DerivedT tmp = d;
        d.m_val      = static_cast<ValueT>(d.m_val + 1);
        return tmp;
    }

    friend constexpr DerivedT operator--(const DerivedT& d, int) noexcept
        requires(EnableInc)
    {
        DerivedT tmp = d;
        d.m_val      = static_cast<ValueT>(d.m_val - 1);
        return tmp;
    }

    friend constexpr bool operator==(const DerivedT a, const DerivedT b) noexcept { return a.m_val == b.m_val; }
    friend constexpr bool operator!=(const DerivedT a, const DerivedT b) noexcept { return a.m_val != b.m_val; }
    friend constexpr bool operator<(const DerivedT a, const DerivedT b) noexcept { return a.m_val < b.m_val; }
    friend constexpr bool operator<=(const DerivedT a, const DerivedT b) noexcept { return a.m_val <= b.m_val; }
    friend constexpr bool operator>(const DerivedT a, const DerivedT b) noexcept { return a.m_val > b.m_val; }
    friend constexpr bool operator>=(const DerivedT a, const DerivedT b) noexcept { return a.m_val >= b.m_val; }

    // public to simplify memcpy/templating semantics
    ValueT m_val = NONE_VALUE;
};


// simple wrapper around an array but indexed by an enum
template <typename Enum, typename T,
          std::enable_if_t<std::is_base_of_v<EnumBase<Enum, typename Enum::ValueT, Enum::COUNT_V, Enum::repr_v,
                                                      Enum::EnableInc_v, Enum::EnableArithmetic_v>,
                                             Enum>,
                           int> = 0>
struct EnumArray
{
    static constexpr std::size_t count = Enum::count();
    static_assert(count > 0, "enum_array requires non-zero count");

    using ValueT     = T;
    using indexT     = std::size_t;
    using ContainerT = std::array<T, count>;

    constexpr ValueT&       operator[](const Enum e) noexcept { return m_data[e.index()]; }
    constexpr const ValueT& operator[](const Enum e) const noexcept { return m_data[e.index()]; }

    constexpr ValueT&                     at(const Enum e) { return m_data.at(e.index()); }
    [[nodiscard]] constexpr const ValueT& at(const Enum e) const { return m_data.at(e.index()); }

    constexpr ValueT&                     at_index(const indexT idx) { return m_data.at(idx); }
    [[nodiscard]] constexpr const ValueT& at_index(const indexT idx) const { return m_data.at(idx); }

    [[nodiscard]] static constexpr indexT size() noexcept { return count; }
    constexpr auto                        begin() noexcept { return m_data.begin(); }
    constexpr auto                        end() noexcept { return m_data.end(); }
    [[nodiscard]] constexpr auto          begin() const noexcept { return m_data.begin(); }
    [[nodiscard]] constexpr auto          end() const noexcept { return m_data.end(); }

    constexpr void fill(const ValueT& v) { m_data.fill(v); }

    ContainerT m_data;
};

static constexpr std::array<std::string_view, 9> file_repr{"a", "b", "c", "d", "e", "f", "g", "h", "-"};

struct File : EnumBase<File, uint8_t, 8, file_repr, true, true>
{
    using base = EnumBase;
    using base::EnumBase;

    [[nodiscard]] static constexpr std::optional<File> from_string(const std::string_view& sv)
    {
        if (sv.size() != 1)
        {
            return std::nullopt;
        }
        if (sv.at(0) == '-')
        {
            return File{NONE_VALUE};
        }
        if (sv.at(0) < 'a' || sv.at(0) > 'h')
        {
            return std::nullopt;
        }
        return File{sv.at(0) - 'a'};
    }
};

constexpr File FILE_A{0};
constexpr File FILE_B{1};
constexpr File FILE_C{2};
constexpr File FILE_D{3};
constexpr File FILE_E{4};
constexpr File FILE_F{5};
constexpr File FILE_G{6};
constexpr File FILE_H{7};
constexpr File NO_FILE{8};

static constexpr std::array<std::string_view, 9> rank_repr = {"1", "2", "3", "4", "5", "6", "7", "8", "-"};

struct Rank : EnumBase<Rank, uint8_t, 8, rank_repr, true, true>
{
    using base = EnumBase;
    using base::EnumBase;

    [[nodiscard]] static constexpr std::optional<Rank> from_string(const std::string_view& sv)
    {
        if (sv.size() != 1 || sv.at(0) < '1' || sv.at(0) > '8')
        {
            return std::nullopt;
        }
        if (sv.at(0) == '-')
        {
            return Rank{NONE_VALUE};
        }
        return Rank{sv.at(0) - '1'};
    }
};

constexpr Rank RANK_1{0};
constexpr Rank RANK_2{1};
constexpr Rank RANK_3{2};
constexpr Rank RANK_4{3};
constexpr Rank RANK_5{4};
constexpr Rank RANK_6{5};
constexpr Rank RANK_7{6};
constexpr Rank RANK_8{7};
constexpr Rank NO_RANK{8};

using Coordinates = std::pair<File, Rank>;

static constexpr std::array<std::string_view, 65> square_repr{
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2", "a3",
    "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4", "a5", "b5",
    "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6", "a7", "b7", "c7",
    "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8", "-"};

struct Square : EnumBase<Square, uint8_t, 64, square_repr, true, true>
{
    using base = EnumBase;
    using base::EnumBase;

    constexpr explicit Square(const Coordinates& coordinates)
        : EnumBase(coordinates.first.index() + (coordinates.second.index() << 3))
    {
    }

    constexpr explicit Square(const File file, const Rank rank) : Square(Coordinates{file, rank}) {}

    [[nodiscard]] constexpr File file() const noexcept { return File{m_val & 7}; }

    [[nodiscard]] constexpr Rank rank() const noexcept { return Rank{m_val >> 3}; }

    [[nodiscard]] constexpr Coordinates coordinates() const noexcept { return {file(), rank()}; }

    [[nodiscard]] constexpr Square flipped_horizontally() const noexcept { return Square{file(), RANK_8 - rank()}; }

    [[nodiscard]] constexpr Square flipped_vertically() const noexcept { return Square{FILE_H - file(), rank()}; }

    [[nodiscard]] static constexpr std::optional<Square> from_string(const std::string_view& sv)
    {
        if (sv.size() == 1 && sv.at(0) == '-')
        {
            return Square{NONE_VALUE};
        }
        if (sv.size() != 2 || sv.at(0) < 'a' || sv.at(0) > 'h' || sv.at(1) < '1' || sv.at(1) > '8')
        {
            return std::nullopt;
        }
        return Square{File{sv.at(0) - 'a'}, Rank{sv.at(1) - '1'}};
    }
};

constexpr Square A1{FILE_A, RANK_1};
constexpr Square A2{FILE_A, RANK_2};
constexpr Square A3{FILE_A, RANK_3};
constexpr Square A4{FILE_A, RANK_4};
constexpr Square A5{FILE_A, RANK_5};
constexpr Square A6{FILE_A, RANK_6};
constexpr Square A7{FILE_A, RANK_7};
constexpr Square A8{FILE_A, RANK_8};

constexpr Square B1{FILE_B, RANK_1};
constexpr Square B2{FILE_B, RANK_2};
constexpr Square B3{FILE_B, RANK_3};
constexpr Square B4{FILE_B, RANK_4};
constexpr Square B5{FILE_B, RANK_5};
constexpr Square B6{FILE_B, RANK_6};
constexpr Square B7{FILE_B, RANK_7};
constexpr Square B8{FILE_B, RANK_8};

constexpr Square C1{FILE_C, RANK_1};
constexpr Square C2{FILE_C, RANK_2};
constexpr Square C3{FILE_C, RANK_3};
constexpr Square C4{FILE_C, RANK_4};
constexpr Square C5{FILE_C, RANK_5};
constexpr Square C6{FILE_C, RANK_6};
constexpr Square C7{FILE_C, RANK_7};
constexpr Square C8{FILE_C, RANK_8};

constexpr Square D1{FILE_D, RANK_1};
constexpr Square D2{FILE_D, RANK_2};
constexpr Square D3{FILE_D, RANK_3};
constexpr Square D4{FILE_D, RANK_4};
constexpr Square D5{FILE_D, RANK_5};
constexpr Square D6{FILE_D, RANK_6};
constexpr Square D7{FILE_D, RANK_7};
constexpr Square D8{FILE_D, RANK_8};

constexpr Square E1{FILE_E, RANK_1};
constexpr Square E2{FILE_E, RANK_2};
constexpr Square E3{FILE_E, RANK_3};
constexpr Square E4{FILE_E, RANK_4};
constexpr Square E5{FILE_E, RANK_5};
constexpr Square E6{FILE_E, RANK_6};
constexpr Square E7{FILE_E, RANK_7};
constexpr Square E8{FILE_E, RANK_8};

constexpr Square F1{FILE_F, RANK_1};
constexpr Square F2{FILE_F, RANK_2};
constexpr Square F3{FILE_F, RANK_3};
constexpr Square F4{FILE_F, RANK_4};
constexpr Square F5{FILE_F, RANK_5};
constexpr Square F6{FILE_F, RANK_6};
constexpr Square F7{FILE_F, RANK_7};
constexpr Square F8{FILE_F, RANK_8};

constexpr Square G1{FILE_G, RANK_1};
constexpr Square G2{FILE_G, RANK_2};
constexpr Square G3{FILE_G, RANK_3};
constexpr Square G4{FILE_G, RANK_4};
constexpr Square G5{FILE_G, RANK_5};
constexpr Square G6{FILE_G, RANK_6};
constexpr Square G7{FILE_G, RANK_7};
constexpr Square G8{FILE_G, RANK_8};

constexpr Square H1{FILE_H, RANK_1};
constexpr Square H2{FILE_H, RANK_2};
constexpr Square H3{FILE_H, RANK_3};
constexpr Square H4{FILE_H, RANK_4};
constexpr Square H5{FILE_H, RANK_5};
constexpr Square H6{FILE_H, RANK_6};
constexpr Square H7{FILE_H, RANK_7};
constexpr Square H8{FILE_H, RANK_8};

constexpr Square NO_SQUARE{64};

static constexpr std::array<std::string_view, 7> piece_type_repr = {"p", "n", "b", "r", "q", "k", "-"};

struct PieceType : EnumBase<PieceType, uint8_t, 6, piece_type_repr, true, true>
{
    using base = EnumBase;
    using base::EnumBase;

    using PieceValueT = int;
    [[nodiscard]] constexpr PieceValueT piece_value() const;
};

constexpr PieceType PAWN{0};
constexpr PieceType KNIGHT{1};
constexpr PieceType BISHOP{2};
constexpr PieceType ROOK{3};
constexpr PieceType QUEEN{4};
constexpr PieceType KING{5};
constexpr PieceType NO_PIECE_TYPE{6};

inline EnumArray<PieceType, PieceType::PieceValueT> piece_type_value{100, 300, 325, 500, 900, 20000};

[[nodiscard]] constexpr PieceType::PieceValueT PieceType::piece_value() const
{
    return piece_type_value.at(*this);
}

static constexpr std::array<std::string_view, 3> color_repr = {"w", "b", "-"};

struct Color : EnumBase<Color, uint8_t, 2, color_repr>
{
    using base = EnumBase;
    using base::EnumBase;
    constexpr Color operator!() const { return Color{m_val ^ 1}; }
    constexpr Color operator~() const { return !(*this); }

    [[nodiscard]] static constexpr std::optional<Color> from_string(const std::string_view& sv)
    {
        if (sv.size() != 1)
            return std::nullopt;
        const char c   = sv[0];
        const int  val = c == 'w' ? 0 : c == 'b' ? 1 : c == '-' ? 2 : 3;
        if (val == 3)
            return std::nullopt;
        return Color{val};
    }
};

constexpr Color WHITE{0};
constexpr Color BLACK{1};
constexpr Color NO_COLOR{2};

static constexpr std::array<std::string_view, 13> piece_repr = {"P", "p", "N", "n", "B", "b", "R",
                                                                "r", "Q", "q", "K", "k", "-"};

struct Piece : EnumBase<Piece, uint8_t, 12, piece_repr, true, true>
{
    using base = EnumBase;
    using base::EnumBase;

    explicit constexpr Piece(const Color c, const PieceType pt) : EnumBase(c.value() + (pt.value() << 1)) {}

    [[nodiscard]] constexpr PieceType type() const { return PieceType{m_val >> 1}; }

    [[nodiscard]] constexpr Color color() const { return Color{m_val & 1}; }

    using PieceValueT = int;
    [[nodiscard]] constexpr PieceValueT piece_value() const { return piece_type_value.at(this->type()); }
};

constexpr Piece W_PAWN{WHITE, PAWN};
constexpr Piece W_KNIGHT{WHITE, KNIGHT};
constexpr Piece W_BISHOP{WHITE, BISHOP};
constexpr Piece W_ROOK{WHITE, ROOK};
constexpr Piece W_QUEEN{WHITE, QUEEN};
constexpr Piece W_KING{WHITE, KING};
constexpr Piece B_PAWN{BLACK, PAWN};
constexpr Piece B_KNIGHT{BLACK, KNIGHT};
constexpr Piece B_BISHOP{BLACK, BISHOP};
constexpr Piece B_ROOK{BLACK, ROOK};
constexpr Piece B_QUEEN{BLACK, QUEEN};
constexpr Piece B_KING{BLACK, KING};
constexpr Piece NO_PIECE{12};

// add a to a square value to get a shift in the associated direction
// shift a bitboard by the value to get a shift in the associated direction
// be careful, this on its own wraps around the board edges
enum Direction
{
    NORTH        = 8,
    EAST         = 1,
    SOUTH        = -NORTH,
    WEST         = -EAST,
    NORTH_EAST   = NORTH + EAST,
    NORTH_WEST   = NORTH + WEST,
    SOUTH_EAST   = SOUTH + EAST,
    SOUTH_WEST   = SOUTH + WEST,
    NO_DIRECTION = 0
};

constexpr Direction direction_from(const Square a, const Square b)
{
    constexpr std::array dir_table{SOUTH_WEST, SOUTH,      SOUTH_EAST, WEST,      NO_DIRECTION,
                                   EAST,       NORTH_WEST, NORTH,      NORTH_EAST};

    // who has time to write 8 if statements
    const int dr = b.rank().value() - a.rank().value();
    const int df = b.file().value() - a.file().value();
    const int nr = (dr > 0) - (dr < 0);
    const int nf = (df > 0) - (df < 0);

    return dir_table.at((nr + 1) * 3 + (nf + 1));
}

constexpr Square operator+(const Square s, const Direction d)
{
    return Square{s.value() + d};
}
constexpr Square operator-(const Square s, const Direction d)
{
    return Square{s.value() - d};
}

template <Direction d>
constexpr auto inverse_dir = static_cast<Direction>(-d);

template <Color c, Direction d>
constexpr Direction relative_dir = c == WHITE ? d : inverse_dir<d>;

template <Color c, Rank r>
constexpr Rank relative_rank = c == WHITE ? r : RANK_8 - r;

template <Color c, File f>
constexpr File relative_file = f;

template <Color c, Square sq>
constexpr auto relative_square = Square{relative_file<c, sq.file()>, relative_rank<c, sq.rank()>};

enum CastlingSide : uint8_t
{
    KINGSIDE  = 0,
    QUEENSIDE = 1,
};

static constexpr std::array<std::string_view, 5> castling_type_repr = {"K", "Q", "k", "q", "-"};

struct CastlingType : EnumBase<CastlingType, uint8_t, 4, castling_type_repr>
{
    using base = EnumBase;
    using base::EnumBase;

    constexpr explicit CastlingType(const Color c, const CastlingSide side) : CastlingType((c.value() << 1) + side) {}

    [[nodiscard]] constexpr Color        color() const { return Color{m_val >> 1}; }
    [[nodiscard]] constexpr CastlingSide side() const { return static_cast<CastlingSide>(m_val & 1); }

    [[nodiscard]] constexpr ValueT mask() const { return m_val == 4 ? 0 : 1 << m_val; }

    [[nodiscard]] constexpr std::pair<Square, Square> king_move() const
    {
        constexpr EnumArray<CastlingType, std::pair<Square, Square>> king_moves{
            std::pair{E1, G1}, {E1, C1}, {E8, G8}, {E8, C8}};
        return king_moves.at(*this);
    }
    [[nodiscard]] constexpr std::pair<Square, Square> rook_move() const
    {
        constexpr EnumArray<CastlingType, std::pair<Square, Square>> rook_moves{
            std::pair{H1, F1}, {A1, D1}, {H8, F8}, {A8, D8}};
        return rook_moves.at(*this);
    }
};

constexpr CastlingType WHITE_KINGSIDE{WHITE, KINGSIDE};
constexpr CastlingType BLACK_KINGSIDE{BLACK, KINGSIDE};
constexpr CastlingType WHITE_QUEENSIDE{WHITE, QUEENSIDE};
constexpr CastlingType BLACK_QUEENSIDE{BLACK, QUEENSIDE};
constexpr CastlingType NO_CASTLING_TYPE{4};

struct Move;

struct CastlingRights
{
    using MaskT                        = uint8_t;
    static constexpr std::size_t NComb = 16;

    constexpr CastlingRights() : m_mask(0) {}

    template <typename int_type, std::enable_if_t<std::is_integral_v<int_type>, int> = 0>
    constexpr explicit CastlingRights(const int_type mask) : m_mask(std::min(mask, static_cast<int_type>(0b1111)))
    {
    }

    constexpr CastlingRights(const std::initializer_list<CastlingType> types) : m_mask(0)
    {
        for (auto t : types)
            m_mask |= t.mask();
    }

    constexpr explicit CastlingRights(const Color c)
        : CastlingRights{CastlingType{c, KINGSIDE}, CastlingType{c, QUEENSIDE}}
    {
    }

    static constexpr std::array<std::string_view, NComb> repr = {"-", "K",  "Q",  "KQ",  "k",  "Kk",  "Qk",  "KQk",
                                                                 "q", "Kq", "Qq", "KQq", "kq", "Kkq", "Qkq", "KQkq"};

    [[nodiscard]] std::string_view to_string() const { return repr.at(m_mask); }

    friend std::ostream& operator<<(std::ostream& os, const CastlingRights& cr)
    {
        os << cr.to_string();
        return os;
    }

    static constexpr CastlingRights all() { return CastlingRights{0b1111}; }
    static constexpr CastlingRights none() { return CastlingRights{0}; }

    friend constexpr bool operator==(const CastlingRights& cr1, const CastlingRights& cr2)
    {
        return cr1.m_mask == cr2.m_mask;
    }
    friend constexpr bool operator!=(const CastlingRights& cr1, const CastlingRights& cr2)
    {
        return cr1.m_mask != cr2.m_mask;
    }

    [[nodiscard]] constexpr bool has(const CastlingType t) const { return m_mask & t.mask(); }

    [[nodiscard]] constexpr bool has_any() const { return m_mask; }
    [[nodiscard]] constexpr bool has_any_color(const Color c) const { return m_mask & CastlingRights(c).m_mask; }

    constexpr void add(const CastlingType t) { m_mask |= t.mask(); }
    constexpr void remove(const CastlingType t) { m_mask &= ~t.mask(); }

    constexpr void remove(const CastlingRights other) { m_mask &= ~other.m_mask; }
    constexpr void keep(const CastlingRights other) { m_mask &= other.m_mask; }

    [[nodiscard]] constexpr bool empty() const { return m_mask == 0; }

    [[nodiscard]] constexpr MaskT mask() const { return m_mask; }

    static const EnumArray<Square, CastlingRights> lost_table;

    [[nodiscard]] constexpr CastlingRights lost_from_move(Move move) const;

    [[nodiscard]] static constexpr std::optional<CastlingRights> from_string(const std::string_view& sv)
    {
        const auto it = std::ranges::find(repr, sv);
        return it == repr.end() ? std::nullopt : std::optional{CastlingRights(std::distance(repr.begin(), it))};
    }


    MaskT m_mask;
};

constexpr EnumArray<Square, CastlingRights> CastlingRights::lost_table = []
{
    EnumArray<Square, CastlingRights> t{};
    for (Square sq = A1; sq <= H8; ++sq)
    {
        t.at(sq) = sq == E1   ? CastlingRights{WHITE_KINGSIDE, WHITE_QUEENSIDE}
                   : sq == H1 ? CastlingRights{WHITE_KINGSIDE}
                   : sq == A1 ? CastlingRights{WHITE_QUEENSIDE}
                   : sq == E8 ? CastlingRights{BLACK_KINGSIDE, BLACK_QUEENSIDE}
                   : sq == H8 ? CastlingRights{BLACK_KINGSIDE}
                   : sq == A8 ? CastlingRights{BLACK_QUEENSIDE}
                              : CastlingRights{};
    }
    return t;
}();

constexpr CastlingRights CASTLING_NONE{NO_CASTLING_TYPE};

constexpr CastlingRights CASTLING_K{WHITE_KINGSIDE};
constexpr CastlingRights CASTLING_Q{WHITE_QUEENSIDE};
constexpr CastlingRights CASTLING_k{BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_q{BLACK_QUEENSIDE};

constexpr CastlingRights CASTLING_KQ{WHITE_KINGSIDE, WHITE_QUEENSIDE};
constexpr CastlingRights CASTLING_Kk{WHITE_KINGSIDE, BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_Kq{WHITE_KINGSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_Qk{WHITE_QUEENSIDE, BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_Qq{WHITE_QUEENSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_kq{BLACK_KINGSIDE, BLACK_QUEENSIDE};

constexpr CastlingRights CASTLING_KQk{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_KQq{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_Kkq{WHITE_KINGSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_Qkq{WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};

constexpr CastlingRights CASTLING_KQkq{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};


constexpr std::array<std::string_view, 4> result_repr = {"1-0", "0-1", "1/2-1/2", "*"};
struct Result : EnumBase<Result, int, 3, result_repr>
{
    using base = EnumBase;
    using base::EnumBase;

    explicit constexpr Result(const Color c) : EnumBase(c.value())
    {}
};

constexpr Result WIN_WHITE{0};
constexpr Result WIN_BLACK{1};
constexpr Result DRAW{2};
constexpr Result NO_RESULT{3};

namespace bit
{
    template <std::unsigned_integral T>
    constexpr int popcount(const T x) noexcept
    {
        return std::popcount(x);
    }

    template <std::unsigned_integral T>
    constexpr int get_lsb(const T bb) noexcept
    {
        return std::countr_zero(bb);
    }

    template <std::unsigned_integral T>
    constexpr int get_msb(const T bb) noexcept
    {
        return std::numeric_limits<T>::digits - 1 - std::countl_zero(bb);
    }

    template <std::unsigned_integral T>
    constexpr int pop_lsb(T& bb) noexcept
    {
        int n = std::countr_zero(bb);
        bb &= ~(static_cast<T>(1) << n);
        return n;
    }

    template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
    constexpr T shift_left(const T value, const unsigned shift)
    {
        // assert(shift < sizeof(T) * 8 && "shift exceeds its bit width");
        return value << shift;
    }

    template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
    constexpr T shift_right(const T value, const unsigned shift)
    {
        // assert(shift < sizeof(T) * 8 && "shift exceeds its bit width");
        return value >> shift;
    }

} // namespace bit


enum move_type_t : uint16_t
{
    NORMAL     = 0,
    PROMOTION  = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING   = 3 << 14
};

// a move is encoded as an 16 bit unsigned int
// 0-5 bit : to square (square 0 to 63)
// 6-11 bit : from square (square 0 to 63)
// 12-13 bit : promotion piece type (shifted by KNIGHT which is the lowest promotion to fit) or
// castle type 14-15: promotion (1), en passant (2), castling (3)
struct Move
{
  public:
    Move() : m_data(0) {}
    constexpr explicit Move(const std::uint16_t d) : m_data(d) {}

    constexpr Move(const Square from, const Square to) : m_data((from.value() << 6) + to.value()) {}

    // to build a move if you already know the type of move
    template <move_type_t T>
    static constexpr Move make(const Square from, const Square to, const PieceType pt = KNIGHT)
    {
        assert(T != CASTLING);
        return Move{static_cast<uint16_t>(T + ((pt - KNIGHT).value() << 12) + (from.value() << 6) + to.value())};
    }

    template <move_type_t T>
    static constexpr Move make(const Square from, const Square to, const CastlingType c)
    {
        assert(T == CASTLING);
        return Move(static_cast<uint16_t>(T + (c.value() << 12) + (from.value() << 6) + to.value()));
    }

    // for sanity check
    [[nodiscard]] constexpr bool is_ok() const { return none().m_data != m_data && null().m_data != m_data; }

    // for these two moves from and to are the same
    static constexpr Move null() { return Move(65); }
    static constexpr Move none() { return Move(0); }

    constexpr bool operator==(const Move& m) const { return m_data == m.m_data; }
    constexpr bool operator!=(const Move& m) const { return m_data != m.m_data; }

    constexpr explicit operator bool() const { return m_data != 0; }

    [[nodiscard]] constexpr std::uint16_t raw() const { return m_data; }

    [[nodiscard]] constexpr Square from_sq() const
    {
        assert(is_ok());
        return Square{(m_data >> 6) & 0x3F};
    }

    [[nodiscard]] constexpr Square to_sq() const
    {
        assert(is_ok());
        return Square{m_data & 0x3F};
    }

    [[nodiscard]] constexpr move_type_t type_of() const { return static_cast<move_type_t>(m_data & 0b11 << 14); }

    [[nodiscard]] constexpr PieceType promotion_type() const { return PieceType{(m_data >> 12 & 0b11)} + KNIGHT; }

    [[nodiscard]] constexpr CastlingType castling_type() const
    {
        return static_cast<CastlingType>(m_data >> 12 & 0b11);
    }

    [[nodiscard]] std::string to_string() const
    {
        std::string s{};
        s.reserve(5);

        s.append(from_sq().to_string());
        s.append(to_sq().to_string());
        if (type_of() == PROMOTION)
        {
            s.append(promotion_type().to_string());
        }
        return s;
    }

    struct UciInfo
    {
        const EnumArray<Square, Piece>& pieces;
        Square ep_square;
        CastlingRights castling_rights;
    };

    static constexpr std::optional<Move> from_uci(const std::string_view& sv, const UciInfo& info);


    friend std::ostream& operator<<(std::ostream& os, const Move mv) { return os << mv.to_string(); }


    struct AlgebraicInfo
    {
        Piece piece;
        bool needs_rank;
        bool needs_file;
        bool is_capture;
        bool is_check;
        bool is_mate;
    };

    std::string to_algebraic(const AlgebraicInfo& info) const
    {
        if (*this == none() || *this == null())
            return "--";

        if (type_of() == CASTLING)
        {
            return castling_type().side() == KINGSIDE ? "O-O" : "O-O-O";
        }

        std::ostringstream oss;

        if (info.piece.type() != PAWN)
            oss << info.piece.type();

        if (info.needs_file)
            oss << from_sq().file();
        if (info.needs_rank)
            oss << from_sq().rank();

        if (info.is_capture)
        {
            if (info.piece.type() == PAWN && !info.needs_file)
                oss << from_sq().file();
            oss << "x";
        }

        oss << to_sq();

        if (type_of() == PROMOTION)
        {
            oss << "=" << Piece{info.piece.color(), promotion_type()}.type();
        }


        if (info.is_check)
        {
            oss << (info.is_mate ? "#" : "+");
        }


        return oss.str();
    }

    std::uint16_t m_data;
};

constexpr std::optional<Move> Move::from_uci(const std::string_view& sv, const UciInfo& info)
{
    if (!(sv.size() == 4 || sv.size() == 5))
        return std::nullopt;

    const auto from = Square::from_string(sv.substr(0, 2));
    const auto to   = Square::from_string(sv.substr(2, 2));

    if (!from || !to)
        return std::nullopt;

    if (sv.size() == 5)
    {
        const auto pt = PieceType::from_string(sv.substr(4, 1));
        if (!pt)
            return std::nullopt;
        return make<PROMOTION>(*from, *to, *pt);
    }

    if (info.pieces.at(*from).type() == PAWN && info.ep_square == *to)
    {
        return make<EN_PASSANT>(*from, *to);
    }

    if (info.pieces.at(*from).type() == KING)
    {
        CastlingRights copy = info.castling_rights;
        while (!copy.empty())
        {
            auto type = CastlingType{bit::get_lsb(copy.mask())};
            if (const auto [k_from, k_to] = type.king_move();
            info.pieces.at(*from).color() == type.color() &&
                *from == k_from && *to == k_to)
            {
                return make<CASTLING>(k_from, k_to, type);
            }
            copy.remove(type);
        }
    }

    return make<NORMAL>(*from, *to);
}

[[nodiscard]] constexpr CastlingRights CastlingRights::lost_from_move(Move move) const
{
    return CastlingRights{(lost_table[move.from_sq()].m_mask | lost_table[move.to_sq()].m_mask) & m_mask};
}

constexpr int MAX_PLY = 255;

constexpr int MATE_SCORE    = 32000;
constexpr int INF_SCORE     = 32001;
constexpr int INVALID_SCORE = 32002;

enum Score : int
{
    MATE             = 32000,
    MATED            = -MATE,
    MATE_IN_MAX_PLY  = MATE - MAX_PLY,
    MATED_IN_MAX_PLY = -MATE_IN_MAX_PLY,
    INF              = 32001,
    INVALID          = 32002
};

constexpr int mate_in(const int ply) noexcept
{
    // better to mate quick -> ply is small
    return MATE - ply;
}

constexpr int mated_in(const int ply) noexcept
{
    // better if mate slow -> ply big
    return MATED + ply;
}

constexpr int absolute_eval(const int eval, const Color side) noexcept
{
    return side == WHITE ? eval : -eval;
}

constexpr int relative_eval(const int eval, const Color side) noexcept
{
    return side == WHITE ? eval : -eval;
}


struct Date {
    int y, m, d;

    [[nodiscard]] std::string to_string() const {
        char buf[11];
        std::sprintf(buf, "%04d.%02d.%02d", y, m, d);
        return buf;
    }
    static bool from_string(const std::string& s, Date& out) {
        if (s.size() != 10) return false;
        int yy, mm, dd;
        if (std::sscanf(s.c_str(), "%d.%d.%d", &yy, &mm, &dd) != 3) return false;

        if (mm < 1 || mm > 12 || dd < 1 || dd > 31) return false;

        out = Date{yy, mm, dd};
        return true;
    }

    static std::optional<Date> from_string(const std::string& s)
    {
        Date d{};
        if (!from_string(s, d)) return std::nullopt;
        return d;
    }
};


inline constexpr auto start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";


#endif
