#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include "types.h"
#include "utils.h"

#include <array>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <hedley.h>
#include <mutex>
#include <random>
#include <ranges>
#include <string>

#define CHEPP_PEXT 0

#if CHEPP_PEXT == 1
#include <immintrin.h>
#endif

namespace chepp {
    inline uint64_t
    pext(const uint64_t val, const uint64_t mask) {
#if CHEPP_PEXT == 1
        return _pext_u64(val, mask);
#endif
        (void)val;
        (void)mask;
        throw std::runtime_error("unsupported pext");
    }

    class Bitboard {
      public:
        using U64 = std::uint64_t;

        static constexpr U64 FILE_A_MASK = 0x0101010101010101ULL;
        static constexpr U64 RANK_1_MASK = 0x00000000000000FFULL;

        static constexpr Bitboard
        empty() noexcept {
            return Bitboard{0};
        }
        static constexpr Bitboard
        full() noexcept {
            return Bitboard{~static_cast<U64>(0)};
        }

        constexpr Bitboard() noexcept = default;
        explicit constexpr Bitboard(const U64 v) noexcept : m_{v} {
        }
        explicit constexpr Bitboard(const Square s) noexcept : m_{static_cast<U64>(1) << s.index()} {
        }
        explicit constexpr Bitboard(const Rank r) noexcept : m_{RANK_1_MASK << (8 * r.index())} {
        }
        explicit constexpr Bitboard(const File f) noexcept : m_{FILE_A_MASK << f.index()} {
        }

        static constexpr Bitboard
        corners() noexcept {
            return Bitboard{A1} | Bitboard{A8} | Bitboard{H1} | Bitboard{H8};
        }
        static constexpr Bitboard
        sides() noexcept {
            return Bitboard{FILE_A} | Bitboard{FILE_H} | Bitboard{RANK_1} | Bitboard{RANK_8};
        }

        [[nodiscard]] explicit constexpr operator U64() const noexcept {
            return m_;
        }
        [[nodiscard]] constexpr U64
        value() const noexcept {
            return m_;
        }

        [[nodiscard]] constexpr Bitboard
        operator~() const noexcept {
            return Bitboard{~m_};
        }
        [[nodiscard]] constexpr Bitboard
        operator|(const Bitboard o) const noexcept {
            return Bitboard{m_ | o.m_};
        }
        [[nodiscard]] constexpr Bitboard
        operator&(const Bitboard o) const noexcept {
            return Bitboard{m_ & o.m_};
        }
        [[nodiscard]] constexpr Bitboard
        operator^(const Bitboard o) const noexcept {
            return Bitboard{m_ ^ o.m_};
        }

        constexpr Bitboard&
        operator|=(const Bitboard o) noexcept {
            m_ |= o.m_;
            return *this;
        }
        constexpr Bitboard&
        operator&=(const Bitboard o) noexcept {
            m_ &= o.m_;
            return *this;
        }
        constexpr Bitboard&
        operator^=(const Bitboard o) noexcept {
            m_ ^= o.m_;
            return *this;
        }

        [[nodiscard]] constexpr Bitboard
        operator<<(const int s) const noexcept {
            return Bitboard{m_ << s};
        }
        [[nodiscard]] constexpr Bitboard
        operator>>(const int s) const noexcept {
            return Bitboard{m_ >> s};
        }

        [[nodiscard]] constexpr bool
        operator==(const Bitboard o) const noexcept {
            return m_ == o.m_;
        }
        [[nodiscard]] constexpr bool
        operator!=(const Bitboard o) const noexcept {
            return m_ != o.m_;
        }
        [[nodiscard]] constexpr explicit operator bool() const noexcept {
            return m_ != 0;
        }

        [[nodiscard]] constexpr bool
        is_set(const int bit) const noexcept {
            return (m_ >> bit) & 1ULL;
        }
        constexpr Bitboard&
        set(const int bit) noexcept {
            m_ |= (1ULL << bit);
            return *this;
        }
        constexpr Bitboard&
        unset(const int bit) noexcept {
            m_ &= ~(1ULL << bit);
            return *this;
        }
        constexpr Bitboard&
        flip(const int bit) noexcept {
            m_ ^= (1ULL << bit);
            return *this;
        }
        [[nodiscard]] constexpr bool
        is_set(const Square sq) const noexcept {
            return is_set(sq.value());
        }
        constexpr Bitboard&
        set(const Square sq) noexcept {
            return set(sq.value());
        }
        constexpr Bitboard&
        unset(const Square sq) noexcept {
            return unset(sq.value());
        }
        constexpr Bitboard&
        flip(const Square sq) noexcept {
            return flip(sq.value());
        }

        [[nodiscard]] constexpr int
        popcount() const noexcept {
            return bit::popcount(m_);
        }
        [[nodiscard]] constexpr int
        get_lsb() const noexcept {
            return bit::get_lsb(m_);
        }
        [[nodiscard]] constexpr int
        pop_lsb() noexcept {
            return bit::pop_lsb(m_);
        }
        [[nodiscard]] constexpr int
        get_msb() const noexcept {
            return bit::get_msb(m_);
        }

        template <typename F>
        void
        for_each_square(F&& f) const {
            Bitboard bb{m_};
            while (bb) {
                f(Square{bb.pop_lsb()});
            }
        }

        class iterator {
          public:
            using value_type        = Square;
            using difference_type   = std::ptrdiff_t;
            using pointer           = Square*;
            using reference         = Square&;
            using iterator_category = std::input_iterator_tag;

            constexpr iterator() noexcept : bits_(0), current_(-1) {
            }
            explicit constexpr iterator(const U64 bits) noexcept : bits_(bits) {
                advance();
            }

            constexpr Square
            operator*() const noexcept {
                return Square{current_};
            }
            constexpr iterator&
            operator++() noexcept {
                advance();
                return *this;
            }
            constexpr iterator
            operator++(int) noexcept {
                const iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            constexpr bool
            operator==(const iterator& o) const noexcept {
                return bits_ == o.bits_ && current_ == o.current_;
            }
            constexpr bool
            operator!=(const iterator& o) const noexcept {
                return !(*this == o);
            }

          private:
            U64 bits_;
            int current_{};

            void constexpr advance() noexcept {
                if (bits_ == 0) {
                    current_ = -1;
                    return;
                }
                current_ = bit::pop_lsb(bits_);
            }
        };

        [[nodiscard]] constexpr iterator
        begin() const noexcept {
            return iterator{m_};
        }
        static constexpr iterator
        end() noexcept {
            return iterator{};
        }

        [[nodiscard]] std::string
        to_string() const {
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

            for (const Square sq : *this) {
                out[row_len + (RANK_8 - sq.rank()).value() * row_len + col_len * (sq.file().value() + 1)] = 'X';
            }

            return out;
        }

        friend std::ostream&
        operator<<(std::ostream& os, const Bitboard& o) {
            os << o.to_string();
            return os;
        }

      private:
        U64 m_{0};
    };

    using bb = Bitboard;
} // namespace chepp

template <>
struct std::hash<chepp::Bitboard> {
    std::size_t
    operator()(const chepp::Bitboard& m) const noexcept {
        return std::hash<std::uint64_t>()(m.value());
    }
};

namespace chepp::movegen {
    namespace detail {
        template <Direction dir>
            requires(dir != NO_DIRECTION)
        static constexpr Bitboard
        direction_mask() {
            if constexpr (dir == EAST || dir == NORTH_EAST || dir == SOUTH_EAST) {
                return ~Bitboard(FILE_H);
            } else if constexpr (dir == WEST || dir == NORTH_WEST || dir == SOUTH_WEST) {
                return ~Bitboard(FILE_A);
            } else {
                return bb::full();
            }
        }
    } // namespace detail

    template <Direction... Dirs>
    static constexpr Bitboard
    shift(Bitboard b) {
        if constexpr (sizeof...(Dirs) == 0) {
            return b;
        } else if constexpr (sizeof...(Dirs) == 1) {
            constexpr Direction dir  = std::get<0>(std::tuple{Dirs...});
            constexpr Bitboard  mask = detail::direction_mask<dir>();
            return dir > 0 ? (b & mask) << dir : (b & mask) >> -dir;
        }
        ((b = shift<Dirs>(b)), ...);
        return b;
    }

    namespace detail {
        template <Direction... Dirs>
        static constexpr Bitboard
        ray(const Square sq, const Bitboard blockers = bb::empty()) {
            if constexpr (sizeof...(Dirs) == 1) {
                constexpr Direction Dir     = std::get<0>(std::tuple{Dirs...});
                Bitboard            attacks = bb::empty();
                Bitboard            bb      = shift<Dir>(Bitboard(sq));
                while (bb) {
                    attacks |= bb;
                    if ((bb & blockers) != bb::empty()) break;
                    bb = shift<Dir>(bb);
                }
                return attacks;
            }
            return (ray<Dirs>(sq, blockers) | ...);
        }

        constexpr Bitboard
        orthogonal_rays(const Square sq, const Bitboard blockers = Bitboard::empty()) {
            return ray<NORTH, EAST, SOUTH, WEST>(sq, blockers);
        }

        constexpr Bitboard
        diagonal_rays(const Square sq, const Bitboard blockers = Bitboard::empty()) {
            return ray<NORTH_EAST, SOUTH_EAST, NORTH_WEST, SOUTH_WEST>(sq, blockers);
        }

        template <PieceType pc>
        constexpr Bitboard
        ray(const Square sq, const Bitboard blockers = bb::empty()) {
            static_assert(pc == BISHOP || pc == ROOK || pc == QUEEN);
            if constexpr (pc == BISHOP)
                return diagonal_rays(sq, blockers);
            else if constexpr (pc == ROOK)
                return orthogonal_rays(sq, blockers);
            else if constexpr (pc == QUEEN)
                return ray<BISHOP>(sq, blockers) | ray<ROOK>(sq, blockers);
            else
                return bb::empty();
        }

        template <PieceType pc>
        constexpr Bitboard
        relevancy_mask(const Square sq) {
            Bitboard mask = ~bb::sides();
            if constexpr (pc == ROOK) {
                if (sq.rank() == RANK_1 || sq.rank() == RANK_8) mask |= Bitboard(sq.rank());
                if (sq.file() == FILE_A || sq.file() == FILE_H) mask |= Bitboard(sq.file());
                mask &= ~bb::corners();
            }
            return ray<pc>(sq) & mask;
        }

        using lines_type = EnumArray<Bitboard, Square, Square>;
        constexpr Bitboard
        compute_lines(const Square sq1, const Square sq2) {
            if (sq1.file() == sq2.file()) return Bitboard{sq1.file()};
            if (sq1.rank() == sq2.rank()) return Bitboard{sq1.rank()};
            if (sq1.file().value() - sq1.rank().value() == sq2.file().value() - sq2.rank().value() ||
                sq1.file().value() + sq1.rank().value() == sq2.file().value() + sq2.rank().value()) {
                return (diagonal_rays(sq1) & diagonal_rays(sq2)).set(sq1).set(sq2);
            }
            return bb::empty();
        }

        using from_to_type = EnumArray<Bitboard, Square, Square>;
        static constexpr Bitboard
        compute_from_to(const Square from, const Square to) {
            return orthogonal_rays(from).is_set(to)
                       ? (orthogonal_rays(from, Bitboard(to)) & orthogonal_rays(to, Bitboard(from))).set(from).set(to)
                   : diagonal_rays(from).is_set(to)
                       ? (diagonal_rays(from, Bitboard(to)) & diagonal_rays(to, Bitboard(from))).set(from).set(to)
                       : bb::empty();
        }

        inline constexpr EnumArray<Bitboard, Color, Square> PAWN_PSEUDO_ATTACKS{
            std::in_place, [](const Color c, const Square sq) {
                const Bitboard bb{sq};
                return c == WHITE ? shift<NORTH_WEST>(bb) | shift<NORTH_EAST>(bb)
                                  : shift<SOUTH_WEST>(bb) | shift<SOUTH_EAST>(bb);
            }};
        inline constexpr EnumArray<Bitboard, PieceType, Square> PIECE_PSEUDO_ATTACKS{
            std::in_place, [](const PieceType pt, const Square sq) {
                const Bitboard bb{sq};
                if (pt == KNIGHT) {
                    return shift<NORTH, NORTH, EAST>(bb) | shift<NORTH, NORTH, WEST>(bb) |
                           shift<SOUTH, SOUTH, EAST>(bb) | shift<SOUTH, SOUTH, WEST>(bb) |
                           shift<EAST, EAST, NORTH>(bb) | shift<EAST, EAST, SOUTH>(bb) | shift<WEST, WEST, NORTH>(bb) |
                           shift<WEST, WEST, SOUTH>(bb);
                } else if (pt == BISHOP) {
                    return diagonal_rays(sq);
                } else if (pt == ROOK) {
                    return orthogonal_rays(sq);
                } else if (pt == QUEEN) {
                    return diagonal_rays(sq) | orthogonal_rays(sq);
                } else if (pt == KING) {
                    return shift<NORTH>(bb) | shift<SOUTH>(bb) | shift<EAST>(bb) | shift<WEST>(bb) |
                           shift<NORTH, EAST>(bb) | shift<NORTH, WEST>(bb) | shift<SOUTH, EAST>(bb) |
                           shift<SOUTH, WEST>(bb);
                }
                return bb::empty();
            }};

        template <PieceType pc>
        constexpr std::size_t
        magics_size() {
            size_t ret = 0;
            for (auto sq = A1; sq <= H8; ++sq) {
                ret += 1ULL << relevancy_mask<pc>(sq).popcount();
            }
            return ret;
        }

        template <PieceType pc, typename Indexer>
            requires(std::is_aggregate_v<Indexer>)
        struct MagicsBase {
            using indexer_t  = Indexer;
            using index_type = Indexer::index_type;
            using mask_type  = Bitboard::U64;

            static constexpr size_t size = magics_size<pc>();

            EnumArray<Indexer, Square>    m_indexers{};
            EnumArray<index_type, Square> m_offsets{};
            std::array<mask_type, size>   m_attacks{};

            void
            init() {
                size_t offset = 0;

                for (Square sq : Square::all()) {
                    const auto mask         = relevancy_mask<pc>(sq);
                    const int  combinations = 1 << mask.popcount();

                    std::vector<mask_type> blockers_vec;
                    blockers_vec.reserve(combinations);
                    for (int comb = 0; comb < combinations; ++comb) {
                        Bitboard bb{0};
                        int      idx = 0;
                        for (const Square s : mask)
                            if (comb & (1 << idx++)) bb.set(s);
                        blockers_vec.push_back(bb.value());
                    }

                    m_indexers.at(sq) = Indexer::make(mask.value(), blockers_vec);
                    m_offsets.at(sq)  = static_cast<index_type>(offset);
                    for (auto& i : blockers_vec) {
                        m_attacks[offset + m_indexers.at(sq).index(i)] = ray<pc>(sq, Bitboard{i}).value();
                    }

                    offset += combinations;
                }
            }

            [[nodiscard]] HEDLEY_ALWAYS_INLINE Bitboard
            attack(Square sq, Bitboard occupancy) const {
                return Bitboard{m_attacks[m_offsets[sq] + m_indexers[sq].index(occupancy.value())]};
            }
        };

        struct ShiftIndexer {
            using index_type = int32_t;
            using mask_type  = Bitboard::U64;
            using magic_type = uint64_t;
            using shift_type = int32_t;

            mask_type  m_mask;
            shift_type m_shift;
            magic_type m_magic;

            static ShiftIndexer
            make(const mask_type mask, const std::vector<mask_type>& blockers) {
                return ShiftIndexer{mask, 64 - bit::popcount(mask), find_magic(mask, blockers)};
            }

            [[nodiscard]] constexpr HEDLEY_ALWAYS_INLINE index_type
            index(const mask_type blockers) const {
                return static_cast<index_type>(((blockers & m_mask) * m_magic) >> m_shift);
            }

            [[nodiscard]] static std::string
            type() {
                return "ShiftIndexer";
            }

          private:
            static constexpr magic_type
            find_magic(const mask_type mask, const std::vector<mask_type>& blockers) {
                constexpr uint64_t MAX_TRIES = 1'000'000;

                for (uint64_t tries = 0; tries < MAX_TRIES; ++tries) {
                    const magic_type candidate = random_magic();
                    bool             fail      = false;
                    std::vector      used(1ULL << bit::popcount(mask), -1);

                    for (size_t i = 0; i < blockers.size(); ++i) {
                        const auto idx =
                            static_cast<index_type>(((blockers[i] & mask) * candidate) >> (64 - bit::popcount(mask)));
                        if (used.at(idx) != -1) {
                            fail = true;
                            break;
                        }
                        used.at(idx) = static_cast<index_type>(i);
                    }

                    if (!fail) return candidate;
                }

                throw std::runtime_error("Failed to find magic number for slider movegen");
            }

            static magic_type
            random_magic() {
                static std::random_device                      rd;
                static std::mt19937_64                         gen(rd());
                static std::uniform_int_distribution<uint64_t> dist{};
                return dist(gen) & dist(gen) & dist(gen);
            }
        };

        struct PEXTIndexer {
            using index_type = uint32_t;
            using mask_type  = Bitboard::U64;

            mask_type m_mask;

            static void
            parse(std::istream& is, PEXTIndexer& res) {
                utils::read<mask_type>(is, res.m_mask);
            }

            static void
            serialize(std::ostream& os, const PEXTIndexer& val) {
                utils::write<mask_type>(os, val.m_mask);
            }

            static constexpr PEXTIndexer
            make(const mask_type mask, const std::vector<mask_type>&) {
                return PEXTIndexer(mask);
            }

            [[nodiscard]] HEDLEY_ALWAYS_INLINE index_type
            index(mask_type blockers) const {
                return static_cast<index_type>(pext(blockers, m_mask));
            }
        };

        template <PieceType pc>
        using Magics = std::conditional_t<CHEPP_PEXT, MagicsBase<pc, PEXTIndexer>, MagicsBase<pc, ShiftIndexer>>;
    } // namespace detail
} // namespace chepp::movegen

#endif
