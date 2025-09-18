#ifndef MOVEGEN_H_INCLUDED
#define MOVEGEN_H_INCLUDED

#include "bitboard.h"
#include "position.h"
#include "types.h"

#include <ranges>

struct ScoredMove
{
    Move move{Move::none()};
    int  score{0};

    bool operator<(const ScoredMove& other) const { return score < other.score; }
    bool operator>(const ScoredMove& other) const { return score > other.score; }
};

struct MoveList
{
    static constexpr size_t max_moves = 256;

    using value_type      = ScoredMove;
    using iterator        = ScoredMove*;
    using const_iterator  = const ScoredMove*;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;

    MoveList() : m_moves(), m_size(0) {}

    void add(const Move& m, const int score = 0)
    {
        assert(m_size < max_moves && "MoveList overflow");
        m_moves[m_size++] = {m, score};
    }
    void add(const ScoredMove& mv)
    {
        assert(m_size < max_moves && "MoveList overflow");
        m_moves[m_size++] = mv;
    }
    void push_back(const Move& m, const int score = 0) { add(m, score); }
    void push_back(const ScoredMove& move) { add(move); }

    reference operator[](const size_type index)
    {
        assert(index < m_size);
        return m_moves[index];
    }
    const_reference operator[](const size_type index) const
    {
        assert(index < m_size);
        return m_moves[index];
    }

    void clear() { m_size = 0; }
    void shrink(const size_type n)
    {
        assert(n <= m_size);
        m_size -= n;
    }

    [[nodiscard]] size_type                  size() const { return m_size; }
    [[nodiscard]] static constexpr size_type capacity() { return max_moves; }
    [[nodiscard]] bool                       empty() const { return m_size == 0; }

    [[nodiscard]] iterator       begin() { return m_moves.data(); }
    [[nodiscard]] iterator       end() { return m_moves.data() + m_size; }
    [[nodiscard]] const_iterator begin() const { return m_moves.data(); }
    [[nodiscard]] const_iterator end() const { return m_moves.data() + m_size; }
    [[nodiscard]] const_iterator cbegin() const { return m_moves.data(); }
    [[nodiscard]] const_iterator cend() const { return m_moves.data() + m_size; }

    [[nodiscard]] reference front()
    {
        assert(!empty());
        return m_moves[0];
    }
    [[nodiscard]] const_reference front() const
    {
        assert(!empty());
        return m_moves[0];
    }
    [[nodiscard]] reference back()
    {
        assert(!empty());
        return m_moves[m_size - 1];
    }
    [[nodiscard]] const_reference back() const
    {
        assert(!empty());
        return m_moves[m_size - 1];
    }

    void sort() { std::ranges::sort(*this, std::greater<>{}); }

    template <typename Pred>
    void filter(Pred pred)
    {
        int idx = 0;
        for (size_type i = 0; i < m_size; ++i)
        {
            if (pred(m_moves[i]))
                m_moves[idx++] = m_moves[i];
        }
        m_size = idx;
    }

  private:
    std::array<ScoredMove, max_moves> m_moves;
    size_type                         m_size;
};

inline void make_all_promotions(MoveList& list, const Square from, const Square to)
{
    list.add(Move::make<PROMOTION>(from, to, QUEEN));
    list.add(Move::make<PROMOTION>(from, to, ROOK));
    list.add(Move::make<PROMOTION>(from, to, BISHOP));
    list.add(Move::make<PROMOTION>(from, to, KNIGHT));
}

template <move_type_t T>
void add_moves_from_bb(MoveList& list, const Bitboard bb, const int delta)
{
    bb.for_each_square([&](const Square to) { list.add(Move::make<T>(to - delta, to)); });
}

inline void add_promotions(MoveList& list, const Bitboard bb, const int delta)
{
    bb.for_each_square([&](const Square to) { make_all_promotions(list, to - delta, to); });
}

template <Color c>
void gen_pawn_moves(const Position& pos, MoveList& list)
{
    constexpr auto up{relative_dir<c, NORTH>};
    constexpr auto down{relative_dir<c, SOUTH>};
    constexpr auto up_right{relative_dir<c, NORTH_EAST>};
    constexpr auto up_left{relative_dir<c, NORTH_WEST>};

    constexpr Bitboard bb_promotion_rank{relative_rank<c, RANK_7>};
    constexpr Bitboard bb_third_rank{relative_rank<c, RANK_3>};
    const Bitboard     check_mask = pos.check_mask(c) == bb::empty() ? bb::full() : pos.check_mask(c);
    const Bitboard     available  = ~pos.occupancy();
    const Bitboard     enemy      = pos.occupancy(~c);
    const Bitboard     pawns      = pos.occupancy(c, PAWN);
    const Bitboard     ep_bb      = pos.ep_square() == NO_SQUARE ? bb::empty() : bb(pos.ep_square());

    // straight
    {
        Bitboard single_push = shift<up>(pawns & ~bb_promotion_rank) & available;
        Bitboard double_push = shift<up>(single_push & bb_third_rank) & available & check_mask;
        single_push &= check_mask;

        add_moves_from_bb<NORMAL>(list, single_push, up);
        add_moves_from_bb<NORMAL>(list, double_push, up + up);
    }
    // promotion
    if (const Bitboard promotions = pawns & bb_promotion_rank)
    {
        Bitboard push       = shift<up>(promotions) & available & check_mask;
        Bitboard take_right = shift<up_right>(promotions) & enemy & check_mask;
        Bitboard take_left  = shift<up_left>(promotions) & enemy & check_mask;

        add_promotions(list, push, up);
        add_promotions(list, take_right, up_right);
        add_promotions(list, take_left, up_left);
    }
    // capture
    {
        Bitboard       capturable = enemy | ep_bb;
        const Bitboard ep_mask    = (check_mask & shift<down>(ep_bb)) ? ep_bb : bb::empty();

        auto handle_capture = [&](Bitboard bb, const int delta)
        {
            bb.for_each_square(
                [&](const Square to)
                {
                    if (to == pos.ep_square())
                        list.add(Move::make<EN_PASSANT>(to - delta, to));
                    else
                        list.add(Move::make<NORMAL>(to - delta, to));
                });
        };

        Bitboard take_right = shift<up_right>(pawns & ~bb_promotion_rank) & capturable & (check_mask | ep_mask);
        Bitboard take_left  = shift<up_left>(pawns & ~bb_promotion_rank) & capturable & (check_mask | ep_mask);

        handle_capture(take_right, up_right);
        handle_capture(take_left, up_left);
    }
}

template <PieceType pc>
void gen_pc_moves(const Position& pos, MoveList& list)
{
    const Color    c = pos.side_to_move();
    const Bitboard check_mask{pos.check_mask(c) == bb::empty() ? bb::full() : pos.check_mask(c)};
    Bitboard       bb{pos.occupancy(c, pc)};

    bb.for_each_square(
        [&](const Square from)
        {
            Bitboard atk{attacks<pc>(from, pos.occupancy()) & ~pos.occupancy(c) & check_mask};
            atk.for_each_square([&](const Square to) { list.add(Move::make<NORMAL>(from, to)); });
        });
}

inline void gen_castling(const Position& pos, MoveList& list)
{
    const Color          c      = pos.side_to_move();
    const CastlingRights rights = pos.castling_rights();

    if (pos.check_mask(c) || !rights.has_any_color(c))
        return;

    for (const auto side : {KINGSIDE, QUEENSIDE})
    {
        if (const auto type = CastlingType{c, side}; rights.has(type))
        {
            auto [k_from, k_to] = type.king_move();
            auto [r_from, r_to] = type.rook_move();
            assert(pos.piece_at(k_from) == Piece(c, KING));
            bool            safe = (from_to_excl(k_from, r_from) & pos.occupancy()) == bb::empty();
            const Direction dir  = direction_from(k_from, k_to);
            assert(dir != NO_DIRECTION);
            for (auto sq = k_from + dir; sq != k_to && safe; sq = sq + dir)
            {
                safe &= !pos.is_attacking_sq(sq, ~c);
            }
            if (safe)
            {
                list.add(Move::make<CASTLING>(k_from, k_to, type));
            }
        }
    }
}

inline void gen_king_moves(const Position& pos, MoveList& list)
{
    const Color    c     = pos.side_to_move();
    const Square   from  = pos.ksq(c);
    const Bitboard moves = attacks<KING>(from, pos.occupancy());

    (moves & (~pos.occupancy() | pos.occupancy(~c))) // unoccupied or capture
        .for_each_square([&](const Square to) { list.add(Move::make<NORMAL>(from, to)); });

    gen_castling(pos, list);
}

template <Color c>
MoveList gen_moves(const Position& pos)
{
    MoveList  list;
    const int n_checkers = pos.checkers(c).popcount();
    assert(n_checkers <= 2);

    if (n_checkers != 2)
    {
        gen_pawn_moves<c>(pos, list);
        gen_pc_moves<BISHOP>(pos, list);
        gen_pc_moves<KNIGHT>(pos, list);
        gen_pc_moves<ROOK>(pos, list);
        gen_pc_moves<QUEEN>(pos, list);
    }
    gen_king_moves(pos, list);
    return list;
}

inline MoveList gen_moves(const Position& pos)
{
    if (pos.side_to_move() == WHITE)
        return gen_moves<WHITE>(pos);
    return gen_moves<BLACK>(pos);
}

inline MoveList gen_legal(const Position& pos)
{
    MoveList moves = gen_moves(pos);
    moves.filter([&](const ScoredMove& mv) { return pos.is_legal(mv.move); });
    return moves;
}

inline MoveList filter_tactical(const Position& pos, const MoveList& list)
{
    MoveList ret;
    std::ranges::copy_if(list, std::back_inserter(ret),
                         [&](const ScoredMove& mv)
                         {
                             return pos.is_occupied(mv.move.to_sq()) || mv.move.type_of() == EN_PASSANT ||
                                    mv.move.type_of() == PROMOTION ||
                                    attacks(pos.piece_type_at(mv.move.from_sq()), mv.move.to_sq(),
                                            pos.occupancy() & ~Bitboard(mv.move.from_sq()), pos.side_to_move())
                                        .is_set(pos.ksq(~pos.side_to_move()));
                         });
    return ret;
}

inline void perft(const Position& prev, const int ply, size_t& out)
{
    MoveList l = gen_legal(prev);

    if (ply == 1)
    {
        out += l.size();
        return;
    }

    for (const auto [move, score] : l)
    {
        Position next{prev};
        next.do_move(move);
        perft(next, ply - 1, out);
    }
}

#endif