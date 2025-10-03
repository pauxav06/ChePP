#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include "types.h"

#include <array>
#include <cassert>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>

#if CHEPP_PEXT == 1
#include <immintrin.h>
inline uint64_t pext(const uint64_t val, const uint64_t mask)
{
    return _pext_u64(val, mask);
}
#endif

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
    constexpr Bitboard&          set(const int bit) noexcept
    {
        m_ |= (1ULL << bit);
        return *this;
    }
    constexpr Bitboard& unset(const int bit) noexcept
    {
        m_ &= ~(1ULL << bit);
        return *this;
    }
    constexpr Bitboard& flip(const int bit) noexcept
    {
        m_ ^= (1ULL << bit);
        return *this;
    }
    [[nodiscard]] constexpr bool is_set(const Square sq) const noexcept { return is_set(sq.value()); }
    constexpr Bitboard&          set(const Square sq) noexcept { return set(sq.value()); }
    constexpr Bitboard&          unset(const Square sq) noexcept { return unset(sq.value()); }
    constexpr Bitboard&          flip(const Square sq) noexcept { return flip(sq.value()); }

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

    class iterator
    {
      public:
        using value_type        = Square;
        using difference_type   = std::ptrdiff_t;
        using pointer           = Square*;
        using reference         = Square&;
        using iterator_category = std::input_iterator_tag;

        constexpr iterator() noexcept : bits_(0), current_(-1) {}
        explicit constexpr iterator(const U64 bits) noexcept : bits_(bits) { advance(); }

        constexpr Square    operator*() const noexcept { return Square{current_}; }
        constexpr iterator& operator++() noexcept
        {
            advance();
            return *this;
        }
        constexpr iterator operator++(int) noexcept
        {
            const iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        constexpr bool operator==(const iterator& o) const noexcept
        {
            return bits_ == o.bits_ && current_ == o.current_;
        }
        constexpr bool operator!=(const iterator& o) const noexcept { return !(*this == o); }

      private:
        U64 bits_;
        int current_{};

        void advance() noexcept
        {
            if (bits_ == 0)
            {
                current_ = -1;
                return;
            }
            current_ = bit::pop_lsb(bits_);
        }
    };

    [[nodiscard]] constexpr iterator begin() const noexcept { return iterator{m_}; }
    static constexpr iterator        end() noexcept { return iterator{}; }

    [[nodiscard]] std::string to_string() const
    {
        static constexpr char empty_board[] = "  A B C D E F G H   \n"
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

        for (const Square sq : *this)
        {
            out[row_len + (RANK_8 - sq.rank()).value() * row_len + col_len * (sq.file().value() + 1)] = 'X';
        }

        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, const Bitboard& o)
    {
        os << o.to_string();
        return os;
    }

    static EnumArray<Bitboard, Square, Square> LINES;
    static Bitboard                            line(const Square sq1, const Square sq2) { return LINES[sq1][sq2]; }
    static bool                                are_aligned(const Square sq1, const Square sq2, const Square sq3)
    {
        return line(sq1, sq2) == line(sq2, sq3);
    }
    static EnumArray<Bitboard, Square, Square> FROM_TO;
    static Bitboard from_to_incl(const Square sq1, const Square sq2) { return FROM_TO[sq1][sq2]; }
    static Bitboard from_to_excl(const Square sq1, const Square sq2)
    {
        return from_to_incl(sq1, sq2).unset(sq1).unset(sq2);
    }

    template <Direction dir>
    static constexpr Bitboard direction_mask()
    {
        static_assert(dir != NO_DIRECTION, "Invalid direction");
        if constexpr (dir == EAST || dir == NORTH_EAST || dir == SOUTH_EAST)
            return ~Bitboard(FILE_H);
        else if constexpr (dir == WEST || dir == NORTH_WEST || dir == SOUTH_WEST)
            return ~Bitboard(FILE_A);
        else
            return full();
    }

    template <Direction... Dirs>
    static constexpr Bitboard shift(Bitboard b)
    {
        if constexpr (sizeof...(Dirs) == 0)
            return b;
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
    static constexpr Bitboard ray(const Square sq, const Bitboard blockers = empty())
    {
        if constexpr (sizeof...(Dirs) == 1)
        {
            constexpr Direction Dir     = std::get<0>(std::tuple{Dirs...});
            Bitboard            attacks = empty();
            Bitboard            bb      = shift<Dir>(Bitboard(sq));
            while (bb)
            {
                attacks |= bb;
                if ((bb & blockers) != empty())
                    break;
                bb = shift<Dir>(bb);
            }
            return attacks;
        }
        return (ray<Dirs>(sq, blockers) | ...);
    }

    static constexpr Bitboard orthogonal_rays(const Square sq, const Bitboard blockers = empty())
    {
        return ray<NORTH, EAST, SOUTH, WEST>(sq, blockers);
    }

    static constexpr Bitboard diagonal_rays(const Square sq, const Bitboard blockers = empty())
    {
        return ray<NORTH_EAST, SOUTH_EAST, NORTH_WEST, SOUTH_WEST>(sq, blockers);
    }

    struct Initialiser
    {
        Initialiser()
        {
            LINES.fill_pred(
                [](const Square sq1, const Square sq2) -> Bitboard
                {
                    if (sq1.file() == sq2.file())
                        return Bitboard{sq1.file()};
                    if (sq1.rank() == sq2.rank())
                        return Bitboard{sq1.rank()};
                    if (sq1.file().value() - sq1.rank().value() == sq2.file().value() - sq2.rank().value() ||
                        sq1.file().value() + sq1.rank().value() == sq2.file().value() + sq2.rank().value())
                    {
                        return (diagonal_rays(sq1) & diagonal_rays(sq2))
                            .set(sq1)
                            .set(sq2);
                    }
                    return empty();
                });

            FROM_TO.fill_pred(
                [](const Square to, const Square from)
                {
                    return orthogonal_rays(from).is_set(to)
                               ? (orthogonal_rays(from, Bitboard(to)) & orthogonal_rays(to, Bitboard(from))).set(from).set(to)
                           : diagonal_rays(from).is_set(to)
                               ? (diagonal_rays(from, Bitboard(to)) & diagonal_rays(to, Bitboard(from))).set(from).set(to)
                               : empty();
                });
        }
    };

  private:
    U64 m_{0};
};

inline EnumArray<Bitboard, Square, Square> Bitboard::LINES   = {};
inline EnumArray<Bitboard, Square, Square> Bitboard::FROM_TO = {};

using bb = Bitboard;

namespace Movegen
{
    namespace detail
    {

        inline EnumArray<Bitboard, Color, Square>     PAWN_PSEUDO_ATTACKS;
        inline EnumArray<Bitboard, PieceType, Square> PIECE_PSEUDO_ATTACKS;

        template <PieceType pc>
        constexpr Bitboard ray(const Square sq, const Bitboard blockers = bb::empty())
        {
            static_assert(pc == BISHOP || pc == ROOK || pc == QUEEN);
            if constexpr (pc == BISHOP)
                return bb::diagonal_rays(sq, blockers);
            else if constexpr (pc == ROOK)
                return bb::orthogonal_rays(sq, blockers);
            else if constexpr (pc == QUEEN)
                return ray<BISHOP>(sq, blockers) | ray<ROOK>(sq, blockers);
            else
                return bb::empty();
        }

    } // namespace detail

    template <PieceType pc>
    constexpr Bitboard pseudo_attack(const Square sq, const Color c)
    {
        static_assert(pc != NO_PIECE_TYPE, "Invalid piece type");
        if constexpr (pc == PAWN)
            return detail::PAWN_PSEUDO_ATTACKS[c][sq];
        else if constexpr (pc == KNIGHT || pc == BISHOP || pc == ROOK || pc == QUEEN || pc == KING)
            return detail::PIECE_PSEUDO_ATTACKS[pc][sq];
        return bb::empty();
    }

    namespace detail
    {

        template <PieceType pc>
        struct Magics
        {
            struct MagicValue
            {
                using index_type = uint32_t;
                using mask_type  = Bitboard;
                mask_type mask;
#if CHEPP_PEXT != 1
                using shift_type = int32_t;
                using magic_type = Bitboard::U64;
                magic_type magic{};
                shift_type shift{};
#endif
                [[nodiscard]] __always_inline index_type operator()(const Bitboard blockers) const
                {
#if CHEPP_PEXT != 1
                    return ((blockers & mask).value() * magic) >> shift;
#else
                    return pext(blockers.value(), mask.value());
#endif
                }
            };

            static constexpr Bitboard relevancy_mask(const Square sq)
            {
                // relevancy mask are squares where blockers are relevant for a piece sliding attack computation. // aka
                // blockers that could stop a ray from a sliding attack // for bishop: it's rays minus the sides because
                // we stop the ray anyway on the side // for rook it's the same except we keep the side if it sits on it
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

            static constexpr std::size_t sz = []()
            {
                size_t ret = 0;
                for (auto sq = A1; sq <= H8; ++sq)
                {
                    ret += 1ULL << relevancy_mask(sq).popcount();
                }
                return ret;
            }();

            void init()
            {
                constexpr uint64_t              MAX_TRIES{UINT64_MAX};
                typename MagicValue::index_type offset{0};

                for (auto sq : Square::all())
                {
                    const typename MagicValue::mask_type mask{relevancy_mask(sq)};
                    const int                            nb_ones{mask.popcount()};
                    const int                            combinations{1 << nb_ones};
#if CHEPP_PEXT != 1
                    const typename MagicValue::shift_type shift{64 - nb_ones};
                    typename MagicValue::magic_type       magic{0};


                    static auto random_magic = []() -> uint64_t
                    {
                        static std::random_device                      rd;
                        static std::mt19937                            gen(rd());
                        static std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
                        return dist(gen) & dist(gen) & dist(gen);
                    };
#endif

                    auto mask_nb = [mask](const uint64_t n) {
                        const Bitboard target_bits{n};
                        Bitboard       bb{0};
                        int            idx = 0;
                        for (const Square sq : mask)
                            if (target_bits.is_set(idx++))
                                bb.set(sq);
                        return bb;
                    };

                    std::vector<Bitboard> cached_blockers(combinations);
                    std::vector<Bitboard> cached_attacks(combinations);

                    for (int comb = 0; comb < combinations; comb++)
                    {
                        cached_blockers[comb] = mask_nb(comb);
                        cached_attacks[comb]  = ray<pc>(sq, cached_blockers[comb]);
                    }

                    offsets.at(sq)         = offset;
                    magic_vals.at(sq).mask = mask;

                    for (uint64_t tries = 1;; tries++)
                    {
                        std::vector<bool> tries_map(combinations, false);
                        bool              fail{false};
#if CHEPP_PEXT != 1
                        magic                   = random_magic();
                        magic_vals.at(sq).magic = magic;
                        magic_vals.at(sq).shift = shift;
#endif

                        for (int c = 0; c < combinations; c++)
                        {
                            const typename MagicValue::index_type index = magic_vals.at(sq)(cached_blockers[c]);

                            if (tries_map[index] && (attacks.at(offset + index) != cached_attacks[c]))
                            {
                                fail = true;
                                break;
                            }
                            attacks.at(offset + index) = cached_attacks[c];
                            tries_map[index]           = true;
                        }

                        if (!fail)
                        {
                            break;
                        }
                        if (tries == MAX_TRIES - 1)
                        {
                            throw std::runtime_error("failed to find magic number for slider movegen");
                        }
                    }
                    offset += combinations;
                }
            }

            [[nodiscard]] Bitboard attack(const Square sq, const Bitboard occupancy) const
            {
                return attacks[offsets[sq] + magic_vals[sq](occupancy)];
            }

            EnumArray<MagicValue, Square>                      magic_vals{};
            EnumArray<typename MagicValue::index_type, Square> offsets{};
            std::array<Bitboard, sz>                           attacks{};
        };

        template <PieceType pc>
        Magics<pc>& magics();

        inline Magics<BISHOP> MAGIC_BISHOPS;
        inline Magics<ROOK>   MAGIC_ROOKS;

        template <>
        inline Magics<BISHOP>& magics<BISHOP>()
        {
            return MAGIC_BISHOPS;
        }

        template <>
        inline Magics<ROOK>& magics<ROOK>()
        {
            return MAGIC_ROOKS;
        }
    } // namespace detail

    template <PieceType pc>
    Bitboard attacks(const Square sq, const Bitboard occupancy = bb::empty(), const Color c = WHITE)
    {
        static_assert(pc, "Invalid piece type");
        if constexpr (pc == ROOK || pc == BISHOP)
        {
            return detail::magics<pc>().attack(sq, occupancy);
        }
        else if constexpr (pc == QUEEN)
        {
            return attacks<BISHOP>(sq, occupancy) | attacks<ROOK>(sq, occupancy);
        }
        else if constexpr (pc == PAWN || pc == KNIGHT || pc == KING)
        {
            return pseudo_attack<pc>(sq, c);
        }
        return {};
    }

    inline Bitboard attacks(const PieceType pt, const Square sq, const Bitboard occupancy = bb::empty(),
                            const Color c = WHITE)
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

    struct Initialiser
    {
        Initialiser()
        {
            detail::PAWN_PSEUDO_ATTACKS.fill_pred(
                [](const Color c, const Square sq)
                {
                    const Bitboard bb{sq};
                    return c == WHITE ? bb::shift<NORTH_WEST>(bb) | bb::shift<NORTH_EAST>(bb)
                                      : bb::shift<SOUTH_WEST>(bb) | bb::shift<SOUTH_EAST>(bb);
                });

            detail::PIECE_PSEUDO_ATTACKS.fill_pred(
                [](const PieceType pt, const Square sq)
                {
                    using namespace Movegen::detail;

                    const Bitboard bb{sq};
                    switch (pt.value())
                    {
                        case KNIGHT.value():
                            return bb::shift<NORTH, NORTH, EAST>(bb) | bb::shift<NORTH, NORTH, WEST>(bb) |
                                   bb::shift<SOUTH, SOUTH, EAST>(bb) | bb::shift<SOUTH, SOUTH, WEST>(bb) |
                                   bb::shift<EAST, EAST, NORTH>(bb) | bb::shift<EAST, EAST, SOUTH>(bb) |
                                   bb::shift<WEST, WEST, NORTH>(bb) | bb::shift<WEST, WEST, SOUTH>(bb);

                        case BISHOP.value():
                            return ray<BISHOP>(sq);

                        case ROOK.value():
                            return ray<ROOK>(sq);

                        case QUEEN.value():
                            return ray<BISHOP>(sq) | ray<ROOK>(sq);

                        case KING.value():
                            return bb::shift<NORTH>(bb) | bb::shift<SOUTH>(bb) | bb::shift<EAST>(bb) |
                                   bb::shift<WEST>(bb) | bb::shift<NORTH, EAST>(bb) | bb::shift<NORTH, WEST>(bb) |
                                   bb::shift<SOUTH, EAST>(bb) | bb::shift<SOUTH, WEST>(bb);

                        default:
                            return Bitboard::empty();
                    }
                });

            detail::MAGIC_BISHOPS.init();
            detail::MAGIC_ROOKS.init();
        }
    };
} // namespace Movegen



#endif
