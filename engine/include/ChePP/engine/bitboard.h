#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED
#include "types.h"
#include <array>
#include <cassert>
#include <cstdlib>
#include <random>
#include <string>

class Bitboard
{
  public:
    using U64 = std::uint64_t;

    static constexpr U64 FILE_A_MASK = 0x0101010101010101ULL;
    static constexpr U64 RANK_1_MASK = 0x00000000000000FFULL;

    static constexpr Bitboard empty() noexcept { return Bitboard{0}; }
    static constexpr Bitboard full() noexcept { return Bitboard{~static_cast<U64>(0)}; }

    constexpr Bitboard() noexcept = default;
    explicit constexpr Bitboard(const U64 v) noexcept : m_{v} {}
    explicit constexpr Bitboard(const Square s) noexcept : m_{static_cast<U64>(1) << s.index()} {}
    explicit constexpr Bitboard(const Rank r) noexcept : m_{RANK_1_MASK << (8 * r.index())} {}
    explicit constexpr Bitboard(const File f) noexcept : m_{FILE_A_MASK << f.index()} {}

    static constexpr Bitboard corners() noexcept { return Bitboard{A1} | Bitboard{A8} | Bitboard{H1} | Bitboard{H8}; }
    static constexpr Bitboard sides() noexcept
    {
        return Bitboard{FILE_A} | Bitboard{FILE_H} | Bitboard{RANK_1} | Bitboard{RANK_8};
    }

    [[nodiscard]] explicit constexpr operator U64() const noexcept { return m_; }
    [[nodiscard]] constexpr U64      value() const noexcept { return m_; }

    // bitwise ops
    [[nodiscard]] constexpr Bitboard operator~() const noexcept { return Bitboard{~m_}; }
    [[nodiscard]] constexpr Bitboard operator|(const Bitboard o) const noexcept { return Bitboard{m_ | o.m_}; }
    [[nodiscard]] constexpr Bitboard operator&(const Bitboard o) const noexcept { return Bitboard{m_ & o.m_}; }
    [[nodiscard]] constexpr Bitboard operator^(const Bitboard o) const noexcept { return Bitboard{m_ ^ o.m_}; }

    constexpr Bitboard& operator|=(const Bitboard o) noexcept
    {
        m_ |= o.m_;
        return *this;
    }
    constexpr Bitboard& operator&=(const Bitboard o) noexcept
    {
        m_ &= o.m_;
        return *this;
    }
    constexpr Bitboard& operator^=(const Bitboard o) noexcept
    {
        m_ ^= o.m_;
        return *this;
    }

    [[nodiscard]] constexpr Bitboard operator<<(const int s) const noexcept { return Bitboard{m_ << s}; }
    [[nodiscard]] constexpr Bitboard operator>>(const int s) const noexcept { return Bitboard{m_ >> s}; }

    // tests
    [[nodiscard]] constexpr bool     operator==(const Bitboard o) const noexcept { return m_ == o.m_; }
    [[nodiscard]] constexpr bool     operator!=(const Bitboard o) const noexcept { return m_ != o.m_; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return m_ != 0; }

    // single bit ops
    [[nodiscard]] constexpr bool is_set(const int bit) const noexcept { return (m_ >> bit) & 1ULL; }
    constexpr void               set(const int bit) noexcept { m_ |= (1ULL << bit); }
    constexpr void               unset(const int bit) noexcept { m_ &= ~(1ULL << bit); }
    constexpr void               flip(const int bit) noexcept { m_ ^= (1ULL << bit); }
    [[nodiscard]] constexpr bool is_set(const Square sq) const noexcept { return is_set(sq.value()); }
    constexpr void               set(const Square sq) noexcept { set(sq.value()); }
    constexpr void               unset(const Square sq) noexcept { unset(sq.value()); }
    constexpr void               flip(const Square sq) noexcept { flip(sq.value()); }

    // popcount/lsb/msb delegated to bit::utils
    [[nodiscard]] constexpr int popcount() const noexcept { return bit::popcount(m_); }
    [[nodiscard]] constexpr int get_lsb() const noexcept { return bit::get_lsb(m_); }
    [[nodiscard]] constexpr int pop_lsb() noexcept { return bit::pop_lsb(m_); }
    [[nodiscard]] constexpr int get_msb() const noexcept { return bit::get_msb(m_); }

    template <typename F>
    void for_each_square(F&& f) const
    {
        Bitboard bb{m_};
        while (bb)
        {
            f(Square{bb.pop_lsb()});
        }
    }


    [[nodiscard]] std::string to_string() const
    {
        static constexpr char empty_board[] =
            "  A B C D E F G H   \n"
            "8 . . . . . . . . 8 \n"
            "7 . . . . . . . . 7 \n"
            "6 . . . . . . . . 6 \n"
            "5 . . . . . . . . 5 \n"
            "4 . . . . . . . . 4 \n"
            "3 . . . . . . . . 3 \n"
            "2 . . . . . . . . 2 \n"
            "1 . . . . . . . . 1 \n"
            "  A B C D E F G H   \n";

        static constexpr auto row_len = std::distance(empty_board, std::ranges::find(empty_board, '8'));
        static constexpr auto col_len = std::distance(empty_board, std::ranges::find(empty_board, 'A'));

        std::string out(empty_board);

        for_each_square([&] (const Square sq){
            out[row_len + (RANK_8 - sq.rank()).value() * row_len + col_len * (sq.file().value() + 1)] = 'X';
        });

        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, const Bitboard& o)
    {
        os << o.to_string();
        return os;
    }


  private:
    U64 m_{0};
};

using bb = Bitboard;

template <Direction dir>
constexpr Bitboard direction_mask()
{
    static_assert(dir != NO_DIRECTION, "Invalid direction");
    if constexpr (dir == EAST || dir == NORTH_EAST || dir == SOUTH_EAST)
        return ~Bitboard(FILE_H);
    else if constexpr (dir == WEST || dir == NORTH_WEST || dir == SOUTH_WEST)
        return ~Bitboard(FILE_A);
    else
        return Bitboard::full();
}


template <Direction... Dirs>
constexpr Bitboard shift(Bitboard b)
{
    if constexpr (sizeof...(Dirs) == 0) return b;
    else if constexpr (sizeof...(Dirs) == 1)
    {
        constexpr Direction dir  = std::get<0>(std::tuple{Dirs...});
        constexpr Bitboard  mask = direction_mask<dir>();
        return dir > 0 ? (b & mask) << dir : (b & mask) >> -dir;
    }
    ((b = shift<Dirs>(b)), ...);
    return b;
}

template <Direction... Dirs>
constexpr Bitboard ray(const Square sq, const Bitboard blockers = bb::empty())
{
    if constexpr (sizeof...(Dirs) == 1)
    {
        constexpr Direction Dir     = std::get<0>(std::tuple{Dirs...});
        Bitboard            attacks = bb::empty();
        Bitboard            bb      = shift<Dir>(Bitboard(sq));
        while (bb)
        {
            attacks |= bb;
            if ((bb & blockers) != bb::empty())
                break;
            bb = shift<Dir>(bb);
        }
        return attacks;
    }
    return (ray<Dirs>(sq, blockers) | ...);
}

template <PieceType pc>
constexpr Bitboard ray(const Square sq, const Bitboard blockers = bb::empty())
{
    static_assert(pc == BISHOP || pc == ROOK || pc == QUEEN);
    if constexpr (pc == BISHOP)
        return ray<NORTH_WEST, NORTH_EAST, SOUTH_WEST, SOUTH_EAST>(sq, blockers);
    else if constexpr (pc == ROOK)
        return ray<NORTH, SOUTH, EAST, WEST>(sq, blockers);
    else if constexpr (pc == QUEEN)
        return ray<BISHOP>(sq, blockers) | ray<ROOK>(sq, blockers);
    else
        return bb::empty();
}


inline EnumArray<Color, EnumArray<Square, Bitboard>>& pawn_pseudo_attacks()
{
    static EnumArray<Color, EnumArray<Square, Bitboard>> g_pawn_pseudo_attacks = [] () {
        EnumArray<Color, EnumArray<Square, Bitboard>> ret{};
        for (auto sq = A1; sq <= H8; ++sq)
        {
            const Bitboard bb{sq};
            ret.at(WHITE).at(sq)   = shift<NORTH_WEST>(bb) | shift<NORTH_EAST>(bb);
            ret.at(BLACK).at(sq)   = shift<SOUTH_WEST>(bb) | shift<SOUTH_EAST>(bb);
        }
        return ret;
    } ();
    return g_pawn_pseudo_attacks;
}


inline EnumArray<PieceType, EnumArray<Square, Bitboard>>& piece_pseudo_attacks()
{
    static EnumArray<PieceType, EnumArray<Square, Bitboard>> g_piece_pseudo_attacks = [] ()
    {
        EnumArray<PieceType, EnumArray<Square, Bitboard>> ret{};
        for (auto sq = A1; sq <= H8; ++sq)
        {
            const Bitboard bb{sq};
            ret.at(KNIGHT).at(sq) = shift<NORTH, NORTH, EAST>(bb) | shift<NORTH, NORTH, WEST>(bb) |
                                                       shift<SOUTH, SOUTH, EAST>(bb) | shift<SOUTH, SOUTH, WEST>(bb) |
                                                       shift<EAST, EAST, NORTH>(bb) | shift<EAST, EAST, SOUTH>(bb) |
                                                           shift<WEST, WEST, NORTH>(bb) | shift<WEST, WEST, SOUTH>(bb);
            ret.at(BISHOP).at(sq) = ray<BISHOP>(sq);
            ret.at(ROOK).at(sq)   = ray<ROOK>(sq);
            ret.at(QUEEN).at(sq)  = ray<BISHOP>(sq) | ray<ROOK>(sq);
            ret.at(KING).at(sq)   = shift<NORTH>(bb) | shift<SOUTH>(bb) | shift<EAST>(bb) |
                                                     shift<WEST>(bb) | shift<NORTH, EAST>(bb) | shift<NORTH, WEST>(bb) |
                                                     shift<SOUTH, EAST>(bb) | shift<SOUTH, WEST>(bb);
        }
        return ret;
    } ();
    return g_piece_pseudo_attacks;
}



template <PieceType pc>
constexpr Bitboard pseudo_attack(const Square sq, const Color c)
{
    static_assert(pc != NO_PIECE_TYPE, "Invalid piece type");
    if constexpr (pc == PAWN)
        return pawn_pseudo_attacks().at(c).at(sq);
    else if constexpr (pc == KNIGHT || pc == BISHOP || pc == ROOK || pc == QUEEN || pc == KING)
        return piece_pseudo_attacks().at(pc).at(sq);
    return bb::empty();
}

inline const EnumArray<Square, EnumArray<Square, Bitboard>>& lines()
{
    static EnumArray<Square, EnumArray<Square, Bitboard>> g_lines = [] {
        EnumArray<Square, EnumArray<Square, Bitboard>> ret{};
        for (Square sq1 = A1; sq1 <= H8; ++sq1) {
            for (Square sq2 = A1; sq2 <= H8; ++sq2) {
                const Bitboard b1{sq1};
                const Bitboard b2{sq2};
                Bitboard line = bb::empty();

                if (sq1.file() == sq2.file())
                    line = Bitboard{sq1.file()};
                else if (sq1.rank() == sq2.rank())
                    line = Bitboard{sq1.rank()};
                else if (sq1.file().value() - sq1.rank().value() == sq2.file().value() - sq2.rank().value() ||
                         sq1.file().value() + sq1.rank().value() == sq2.file().value() + sq2.rank().value()) {
                    line = (ray<BISHOP>(sq1) & ray<BISHOP>(sq2)) | b1 | b2;
                         }

                ret.at(sq1).at(sq2) = line;
            }
        }
        return ret;
    }();
    return g_lines;
}

inline Bitboard line(const Square sq1, const Square sq2)
{
    return lines().at(sq1).at(sq2);
}

inline bool are_aligned(const Square sq1, const Square sq2, const Square sq3)
{
    return line(sq1, sq2) == line(sq2, sq3);
}

inline const EnumArray<Square, EnumArray<Square, Bitboard>>& from_to()
{
    static EnumArray<Square, EnumArray<Square, Bitboard>> g_from_to = [] {
        EnumArray<Square, EnumArray<Square, Bitboard>> ret{};
        for (auto sq1 = A1; sq1 <= H8; ++sq1)
        {
            for (auto sq2 = A1; sq2 <= H8; ++sq2)
            {
                if (ray<ROOK>(sq1) & Bitboard(sq2))
                {
                    ret.at(sq1).at(sq2) = ray<ROOK>(sq1, Bitboard(sq2)) & ray<ROOK>(sq2, Bitboard(sq1));
                    ret.at(sq1).at(sq2) |= (Bitboard(sq1) | Bitboard(sq2));
                }
                if (ray<BISHOP>(sq1) & Bitboard(sq2))
                {
                    ret.at(sq1).at(sq2) = ray<BISHOP>(sq1, Bitboard(sq2)) & ray<BISHOP>(sq2, Bitboard(sq1));
                    ret.at(sq1).at(sq2) |= (Bitboard(sq1) | Bitboard(sq2));
                }
            }
        }
        return ret;
    }();
    return g_from_to;
}


inline Bitboard from_to_incl(const Square sq1, const Square sq2)
{
    return from_to().at(sq1).at(sq2);
}

inline Bitboard from_to_excl(const Square sq1, const Square sq2)
{
    return from_to_incl(sq1, sq2) & ~Bitboard(sq1) & ~Bitboard(sq2);
}

template <PieceType pc>
constexpr Bitboard relevancy_mask(const Square sq)
{
    // relevancy mask are squares where blockers are relevant for a piece sliding attack computation. // aka blockers
    // that could stop a ray from a sliding attack // for bishop: it's rays minus the sides because we stop the ray
    // anyway on the side // for rook it's the same except we keep the side if it sits on it
    Bitboard mask = ~bb::sides();
    if constexpr (pc == ROOK)
    {
        if (sq.rank() == RANK_1 || sq.rank() == RANK_8)
        {
            mask |= Bitboard(sq.rank());
        }
        if (sq.file() == FILE_A || sq.file() == FILE_H)
        {
            mask |= Bitboard(sq.file());
        }
        mask &= ~bb::corners();
    }
    return ray<pc>(sq) & mask;
}

template <PieceType pc>
struct magics_t
{
    struct magic_val_t
    {
        using shift_type = int32_t;
        using index_type = uint32_t;
        using magic_type = Bitboard::U64;
        using mask_type  = Bitboard;
        mask_type                mask;
        magic_type               magic{};
        shift_type               shift{};
        index_type               offset{};
        [[nodiscard]] index_type index(const Bitboard blockers) const
        {
            return offset + (((blockers & mask).value() * magic) >> shift);
        }
    };
    static constexpr std::size_t sz = [] ()
    {
        size_t ret = 0;
        for (auto sq = A1; sq <= H8; ++sq)
        {
            ret += 1ULL << relevancy_mask<pc>(sq).popcount();
        }
        return ret;
    } ();

    magics_t();

    [[nodiscard]] Bitboard attack(const Square sq, const Bitboard occupancy) const
    {
        return attacks[magic_vals[sq].index(occupancy)];
    }
    EnumArray<Square, magic_val_t> magic_vals;
    std::array<Bitboard, sz>       attacks;
};

template <PieceType pc>
const magics_t<pc>& magics() {
    static magics_t<pc> instance;
    return instance;
}

template <>
inline const magics_t<BISHOP>& magics<BISHOP>() {
    static magics_t<BISHOP> instance;
    return instance;
}

template <>
inline const magics_t<ROOK>& magics<ROOK>() {
    static magics_t<ROOK> instance;
    return instance;
}

static uint64_t       random_u64()
{
    static std::random_device                      rd;
    static std::mt19937                            gen(rd());
    static std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    return dist(gen);
}

static uint64_t random_magic()
{
    return random_u64() & random_u64() & random_u64();
}

constexpr Bitboard mask_nb(const Bitboard mask, const uint64_t n)
{
    Bitboard bb{0};
    int      idx = 0;
    for (auto sq = A1; sq <= H8; ++sq)
    {
        if (Bitboard(sq) & mask)
        {
            if (Bitboard{n}.is_set(idx++))
            {
                bb |= Bitboard(sq);
            }
        }
    }
    return bb;
}

template <PieceType pc>
magics_t<pc>::magics_t()
{
    constexpr uint64_t               MAX_TRIES{UINT64_MAX};
    constexpr uint64_t               MAX_COMB{4096};
    std::array<Bitboard, MAX_COMB>   cached_blockers{};
    std::array<Bitboard, MAX_COMB>   cached_attacks{};
    typename magic_val_t::index_type offset{0};
    for (auto sq = A1; sq <= H8; ++sq)
    {
        const typename magic_val_t::mask_type  mask{relevancy_mask<pc>(sq)};
        const int                              nb_ones{mask.popcount()};
        const int                              combinations{1 << nb_ones};
        const typename magic_val_t::shift_type shift{64 - nb_ones};
        typename magic_val_t::magic_type       magic{0};
        assert(combinations <= MAX_COMB && "to many blockers variations, check relevancy mask");
        for (int comb = 0; comb < combinations; comb++)
        {
            cached_blockers.at(comb) = mask_nb(mask, comb);
            cached_attacks.at(comb)  = ray<pc>(sq, cached_blockers.at(comb));
        }
        for (uint64_t tries = 1;; tries++)
        {
            std::array<bool, MAX_COMB> tries_map{};
            bool                       fail{false};
            magic = random_magic();
            for (int c = 0; c < combinations; c++)
            {
                const typename magic_val_t::index_type index = (cached_blockers.at(c).value() * magic) >> shift;
                if (const Bitboard cur = attacks.at(offset + index);
                    tries_map.at(index) && (cur != cached_attacks.at(c)))
                {
                    fail = true;
                    break;
                }
                attacks.at(offset + index) = cached_attacks.at(c);
                tries_map.at(index)        = true;
            }
            if (!fail)
            {
                break;
            }
            if (tries == MAX_TRIES - 1)
            {
                assert(0 && "failed to find magic");
            }
        }
        magic_vals.at(sq).offset = offset;
        magic_vals.at(sq).magic  = magic;
        magic_vals.at(sq).mask   = mask;
        magic_vals.at(sq).shift  = shift;
        offset += combinations;
    }
}

template <PieceType pc>
Bitboard attacks(const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE)
{
    if constexpr (pc == ROOK || pc == BISHOP)
    {
        return magics<pc>().attack(sq, occupancy);
    }
    else if constexpr (pc == QUEEN)
    {
        return attacks<BISHOP>(sq, occupancy) | attacks<ROOK>(sq, occupancy);
    }
    else if constexpr (pc == PAWN || pc == KNIGHT || pc == KING)
    {
        return pseudo_attack<pc>(sq, c);
    }
    assert(false && "invalid piece type");
    return bb::empty();
}

inline Bitboard attacks(const PieceType pt, const Square sq, const Bitboard occupancy = bb::empty(), const Color c =  WHITE)
{
    switch (pt.value())
    {
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

struct InitTables {
    InitTables() {
        (void)pawn_pseudo_attacks();
        (void)piece_pseudo_attacks();
        (void)lines();
        (void)from_to();
        (void)magics<BISHOP>();
        (void)magics<ROOK>();
    }
};

static InitTables init_tables_instance;

#endif
