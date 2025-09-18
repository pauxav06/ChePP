#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED
#include "bitboard.h"
#include "types.h"
#include "zobrist.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <ostream>
#include <ranges>
#include <src/tbprobe.h>
#include <sstream>
#include <unordered_map>
#include <utility>

struct Position
{
    Position() = default;
    Position(const Position& prev, const Move move) : Position(prev) { do_move(move); }


    void init_zobrist();


    [[nodiscard]] Square                          ep_square() const { return m_ep_square; }
    [[nodiscard]] Piece                           captured() const { return m_captured; }
    [[nodiscard]] Move                            move() const { return m_move; }
    [[nodiscard]] Piece                           moved() const { return m_moved; }
    [[nodiscard]] Color                           side_to_move() const { return m_color; }
    [[nodiscard]] int                             halfmove_clock() const { return m_halfmove_clock; }
    [[nodiscard]] int                             full_move_clock() const { return m_fullmove_clock; }
    [[nodiscard]] CastlingRights                  castling_rights() const { return m_crs; }
    [[nodiscard]] hash_t                          hash() const { return m_hash.value(); }
    [[nodiscard]] const EnumArray<Square, Piece>& pieces() const { return m_pieces; }
    [[nodiscard]] Piece                           piece_at(const Square sq) const { return m_pieces.at(sq); }
    [[nodiscard]] PieceType                       piece_type_at(const Square sq) const { return piece_at(sq).type(); }
    [[nodiscard]] Color                           color_at(const Square sq) const { return piece_at(sq).color(); }
    [[nodiscard]] Square                          ksq(const Color c) const { return m_ksq.at(c); }

    [[nodiscard]] Bitboard checkers(const Color c) const { return m_check_mask.at(c) & occupancy(~c); }
    [[nodiscard]] Bitboard blockers(const Color c) const { return m_blockers.at(c); }
    [[nodiscard]] Bitboard check_mask(const Color c) const { return m_check_mask.at(c); }


    [[nodiscard]] Bitboard occupancy() const { return m_global_occupancy; }
    [[nodiscard]] Bitboard occupancy(const Color c) const { return m_color_occupancy.at(c); }
    [[nodiscard]] Bitboard occupancy(const PieceType p) const { return m_pieces_type_occupancy.at(p); }
    [[nodiscard]] Bitboard occupancy(const Color c, const PieceType p) const { return occupancy(p) & occupancy(c); }
    template <class... Ts>
    [[nodiscard]] Bitboard occupancy(Color c, PieceType first, const Ts... rest) const;
    template <class... Ts>
    [[nodiscard]] Bitboard occupancy(PieceType first, const Ts... rest) const;
    [[nodiscard]] Bitboard occupancy(Color c, std::initializer_list<PieceType> types) const;
    [[nodiscard]] Bitboard occupancy(std::initializer_list<PieceType> types) const;
    [[nodiscard]] bool     is_occupied(const Square sq) const { return m_pieces.at(sq) != NO_PIECE; }


    [[nodiscard]] Bitboard attacking_sq(Square sq, Bitboard occ) const;
    [[nodiscard]] Bitboard attacking_sq(Square sq) const;
    [[nodiscard]] bool     is_attacking_sq(Square sq, Color c) const;


    template <Color c>
    [[nodiscard]] bool is_legal(Move move) const;
    [[nodiscard]] bool is_legal(Move move) const;
    void               do_move(Move move);


    template <PieceType pt>
    void update_checkers_and_blockers(Color c);
    void update_checkers_and_blockers(Color c);
    void update();


    void set_piece(Piece piece, Square sq);
    void set_piece(PieceType piece_type, Color color, Square sq);
    void remove_piece(Square sq);
    void move_piece(Square from, Square to);


    [[nodiscard]] std::string to_string() const;
    friend std::ostream&      operator<<(std::ostream& os, const Position& pos) { return os << pos.to_string(); }
    bool                      from_fen(std::string_view fen);
    [[nodiscard]] std::string to_fen() const;


    [[nodiscard]] unsigned wdl_probe() const;
    [[nodiscard]] unsigned dtz_probe() const;


    [[nodiscard]] int see(Move move) const;
    bool              is_insufficient_material() const;

  private:
    // copied
    zobrist_t                      m_hash{};
    EnumArray<Square, Piece>       m_pieces{};
    EnumArray<Color, Bitboard>     m_color_occupancy{};
    Bitboard                       m_global_occupancy{};
    EnumArray<PieceType, Bitboard> m_pieces_type_occupancy{};
    EnumArray<Color, Square>       m_ksq{};
    CastlingRights                 m_crs{};
    Color                          m_color{};
    uint8_t                        m_halfmove_clock = 0;
    uint8_t                        m_fullmove_clock = 1;
    Square                         m_ep_square{};
    Piece                          m_captured{};
    Move                           m_move{};
    Piece                          m_moved{};
    // 5 available

    // recomputed
    EnumArray<Color, Bitboard> m_blockers{};
    EnumArray<Color, Bitboard> m_check_mask{};
};



inline void Position::init_zobrist()
{
    m_hash = zobrist_t{};
    for (auto sq = A1; sq <= H8; sq = ++sq)
    {
        if (piece_at(sq) != NO_PIECE)
            m_hash.flip_piece(piece_at(sq), sq);
    }

    if (ep_square() != NO_SQUARE)
        m_hash.flip_ep(ep_square().file());

    if (side_to_move() == BLACK)
        m_hash.flip_color();

    m_hash.flip_castling_rights(castling_rights().mask());
}




inline Bitboard Position::occupancy(const Color c, const std::initializer_list<PieceType> types) const
{
    return occupancy(types) & occupancy(c);
}

template <typename... Ts>
[[nodiscard]] Bitboard Position::occupancy(const Color c, const PieceType first, const Ts... rest) const
{
    return occupancy(first, rest...) & occupancy(c);
}

template <typename... Ts>
Bitboard Position::occupancy(const PieceType first, const Ts... rest) const
{
    if constexpr (sizeof...(rest) == 0)
    {
        return occupancy(first);
    }
    else
    {
        return occupancy(first) | occupancy(rest...);
    }
}

inline Bitboard Position::occupancy(const std::initializer_list<PieceType> types) const
{
    Bitboard result{0};

    for (const PieceType pt : types)
        result |= occupancy(pt);

    return result;
}



inline Bitboard Position::attacking_sq(const Square sq, const Bitboard occ) const
{
    // if a piece attacks a square sq2 from a square sq1 it would attack sq1 from sq2
    // exception made for pawns where this is true only with reversed colors
    return ((attacks<ROOK>(sq, occ) & occupancy(ROOK, QUEEN)) | (attacks<BISHOP>(sq, occ) & occupancy(BISHOP, QUEEN)) |
            (attacks<KNIGHT>(sq, occ) & occupancy(KNIGHT)) | (attacks<PAWN>(sq, occ, BLACK) & occupancy(WHITE, PAWN)) |
            (attacks<PAWN>(sq, occ, WHITE) & occupancy(BLACK, PAWN)) | (attacks<KING>(sq, occ) & occupancy(KING))) &
           occ;
}

inline Bitboard Position::attacking_sq(const Square sq) const
{
    return attacking_sq(sq, occupancy());
}

inline bool Position::is_attacking_sq(const Square sq, const Color c) const
{
    // if a piece attacks a square sq2 from a square sq1 it would attack sq1 from sq2
    // exception made for pawns where this is true only with reversed colors
    return attacks<KNIGHT>(sq, occupancy()) & occupancy(c, KNIGHT) ||
           attacks<PAWN>(sq, occupancy(), ~c) & occupancy(c, PAWN) ||
           attacks<KING>(sq, occupancy()) & occupancy(c, KING) ||
           attacks<BISHOP>(sq, occupancy()) & occupancy(c, BISHOP, QUEEN) ||
           attacks<ROOK>(sq, occupancy()) & occupancy(c, ROOK, QUEEN);
}




template <PieceType pt>
void Position::update_checkers_and_blockers(const Color c)
{
    static_assert(pt == BISHOP || pt == ROOK);
    auto enemies = occupancy(~c, pt, QUEEN);
    enemies.for_each_square(
        [&](const Square sq)
        {
            const auto line       = from_to_excl(sq, ksq(c)) & attacks<pt>(sq);
            const auto on_line    = line & occupancy();
            const auto n_blockers = on_line.popcount();
            if (n_blockers == 0)
            {
                // check mask also contains all squares between the long range attacker and the king
                m_check_mask.at(c) |= line;
            }
            if (n_blockers == 1)
            {
                // blockers can be of any color
                // blocker from same color are pinned from other color can do a discovered check
                m_blockers.at(c) |= on_line;
            }
        });
}

inline void Position::update_checkers_and_blockers(const Color c)
{
    m_blockers.at(c)   = bb::empty();
    m_check_mask.at(c) = attacking_sq(ksq(c)) & occupancy(~c);

    update_checkers_and_blockers<BISHOP>(c);
    update_checkers_and_blockers<ROOK>(c);
}

inline void Position::update()
{
    m_global_occupancy = occupancy(WHITE) | occupancy(BLACK);
    m_ksq              = {Square{occupancy(WHITE, KING).get_lsb()}, Square{occupancy(BLACK, KING).get_lsb()}};

    update_checkers_and_blockers(side_to_move());
    update_checkers_and_blockers(~side_to_move());
}



inline void Position::set_piece(const Piece piece, const Square sq)
{
    assert(!is_occupied(sq));
    const PieceType pt = piece.type();
    const Color     c  = piece.color();
    m_pieces_type_occupancy.at(pt) |= Bitboard(sq);
    m_color_occupancy.at(c) |= Bitboard(sq);
    m_pieces.at(sq) = piece;
}

inline void Position::set_piece(const PieceType piece_type, const Color color, const Square sq)
{
    assert(!is_occupied(sq));

    m_pieces_type_occupancy.at(piece_type) |= Bitboard(sq);
    m_color_occupancy.at(color) |= Bitboard(sq);
    m_pieces.at(sq) = Piece{color, piece_type};
}

inline void Position::remove_piece(const Square sq)
{
    const Piece pc = piece_at(sq);
    m_pieces_type_occupancy.at(pc.type()) &= ~Bitboard(sq);
    m_color_occupancy.at(pc.color()) &= ~Bitboard(sq);
    m_pieces.at(sq) = NO_PIECE;
}

inline void Position::move_piece(const Square from, const Square to)
{
    const Piece piece = piece_at(from);
    remove_piece(from);
    set_piece(piece, to);
}



inline bool Position::from_fen(const std::string_view fen)
{
    std::memset(this, 0, sizeof(*this));
    m_pieces.fill(NO_PIECE);

    std::istringstream iss{std::string(fen)};
    std::string        board_str, color_str, castling_str, ep_str, halfmove_str, fullmove_str;
    iss >> board_str >> color_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;

    File file = FILE_A;
    Rank rank = RANK_8;

    for (const auto c : board_str)
    {
        if (c == '/')
        {
            --rank;
            file = FILE_A;
        }
        else if (c == '8')
        {
        }
        else if (std::isdigit(c))
        {
            file = file + (c - '0');
        }
        else
        {
            const auto pc = Piece::from_string(std::string{c});
            if (!pc)
                return false;
            set_piece(pc.value(), Square{file, rank});

            if (file != FILE_H)
                ++file;
        }
    }

    const auto color = Color::from_string(color_str);
    if (!color)
        return false;
    m_color = color.value();

    const auto crs = CastlingRights::from_string(castling_str);
    if (!crs)
        return false;
    m_crs = crs.value();

    const auto ep_square = Square::from_string(ep_str);
    if (!ep_square)
        return false;
    m_ep_square = ep_square.value();

    if (halfmove_str.size() > 3 || fullmove_str.size() > 3)
        return false;

    try
    {
        m_halfmove_clock = stoi(halfmove_str);
        m_fullmove_clock = stoi(fullmove_str);
    }
    catch (const std::invalid_argument& _)
    {
        return false;
    }
    catch (const std::out_of_range& _)
    {
        return false;
    }

    init_zobrist();
    update();
    return true;
}

inline std::string Position::to_fen() const
{
    std::ostringstream oss;

    for (auto r = RANK_1; r <= RANK_8; ++r)
    {
        const auto rank  = RANK_8 - r;
        int        empty = 0;
        for (auto file = FILE_A; file <= FILE_H; ++file)
        {
            const Square sq{file, rank};

            if (Piece pc = piece_at(sq); pc == NO_PIECE)
            {
                ++empty;
            }
            else
            {
                if (empty > 0)
                {
                    oss << empty;
                    empty = 0;
                }
                oss << pc;
            }
        }
        if (empty > 0)
            oss << empty;

        if (rank > RANK_1)
            oss << '/';
    }

    oss << ' ' << side_to_move();
    oss << ' ' << castling_rights();
    oss << ' ' << ep_square();
    oss << ' ' << halfmove_clock();
    oss << ' ' << full_move_clock();

    return oss.str();
}

inline std::string Position::to_string() const
{
    std::ostringstream res{};

    res << "Position:\n";
    res << "Side to move: " << side_to_move() << "\n";
    res << "Castling rights: " << castling_rights() << "\n";

    res << "En passant square: " << ep_square() << "\n";

    res << "Zobrist hash: 0x" << std::hex << hash() << std::dec << "\n";

    for (auto rank = RANK_1; rank <= RANK_8; ++rank)
    {
        res << (RANK_8 - rank).index() + 1 << " ";
        for (auto file = FILE_A; file <= FILE_H; ++file)
        {
            const Square sq{file, RANK_8 - rank};
            const Piece  pc = piece_at(sq);
            res << pc << " ";
        }
        res << "\n";
    }

    res << "  a b c d e f g h\n";
    return res.str();
}




template <Color c>
bool Position::is_legal(const Move move) const
{
    constexpr Direction down    = c == WHITE ? SOUTH : NORTH;
    const auto          from_bb = Bitboard(move.from_sq());
    const auto          to_bb   = Bitboard(move.to_sq());

    if (ksq(c) == move.from_sq())
    {
        // cannot move along the ray of a long range piece
        Bitboard long_range = checkers(c) & occupancy(~c, ROOK, BISHOP, QUEEN);
        while (long_range)
        {
            if (const auto sq = Square{long_range.pop_lsb()};
                are_aligned(sq, move.from_sq(), move.to_sq()) && move.to_sq() != sq)
            {
                return false;
            }
        }
        // cannot move to a square that is attacked
        if (is_attacking_sq(move.to_sq(), ~c))
        {
            return false;
        }
    }
    else if (move.type_of() == NORMAL || move.type_of() == PROMOTION)
    {
        // check for pins
        // if the piece is a blocker
        if (Bitboard(move.from_sq()) & blockers(c))
        {
            // and if it is not moving along the line it is pinned to
            if (!(Bitboard(move.to_sq()) & line(move.from_sq(), ksq(c))))
            {
                // we are breaking the pin, move is illegal
                return false;
            }
        }
    }
    else if (move.type_of() == EN_PASSANT)
    {
        // en passant can create discovered checks so we check for any long range attack
        // we know they have not moved. therefore we only need to update global occupancy to cast rays
        // we check if rays intersect with any long range and if they do that means a there is a check
        if (const Bitboard ep_occupancy = (occupancy() & ~from_bb | to_bb) & ~shift<down>(to_bb);
            occupancy(~c, BISHOP, QUEEN) & attacks(BISHOP, ksq(c), ep_occupancy) ||
            occupancy(~c, ROOK, QUEEN) & attacks(ROOK, ksq(c), ep_occupancy))
            return false;
    }
    return true;
}

inline bool Position::is_legal(const Move move) const
{
    if (side_to_move() == WHITE)
        return is_legal<WHITE>(move);
    if (side_to_move() == BLACK)
        return is_legal<BLACK>(move);
    return false;
}

inline void Position::do_move(const Move move)
{

    m_hash.flip_color();

    if (ep_square() != NO_SQUARE)
    {
        m_hash.flip_ep(ep_square().file());
    }

    m_halfmove_clock++;
    m_color = ~m_color;
    m_fullmove_clock += m_color == BLACK;
    m_ep_square = NO_SQUARE;
    m_captured  = NO_PIECE;
    m_move      = move;
    m_moved     = NO_PIECE;

    if (move == Move::null())
    {
        update();
        return;
    }

    m_moved = piece_at(move.from_sq());

    const Square from = move.from_sq();
    const Square to   = move.to_sq();
    Piece        pc   = piece_at(from);
    const Color  us   = pc.color();

    assert(us == ~side_to_move());

    const Direction up = us == WHITE ? NORTH : SOUTH;

    const auto lost = m_crs.lost_from_move(move);
    m_crs.remove(lost);
    m_hash.flip_castling_rights(lost.mask());

    if (move.type_of() == CASTLING)
    {
        const auto castling_type = move.castling_type();
        assert(CastlingRights{lost}.has(castling_type));
        auto [k_from, k_to] = castling_type.king_move();
        auto [r_from, r_to] = castling_type.rook_move();

        move_piece(r_from, r_to);
        move_piece(k_from, k_to);

        m_hash.move_piece(Piece{us, KING}, k_from, k_to);
        m_hash.move_piece(Piece{us, ROOK}, r_from, r_to);

        update();
        return;
    }

    if (pc.type() == PAWN || is_occupied(to))
    {
        m_halfmove_clock = 0;
    }

    if (move.type_of() == NORMAL || move.type_of() == PROMOTION)
    {

        // capture
        if (is_occupied(to))
        {
            assert(color_at(to) == ~us);

            m_captured = piece_at(to);
            m_hash.flip_piece(piece_at(to), to);
            remove_piece(to);
        }
        // set new ep square
        else if (pc.type() == PAWN && to.value() - from.value() == 2 * up)
        {
            // only set if ep is actually playable
            if (((pseudo_attack<PAWN>(to - up, us)) & occupancy(~us)) != bb::empty())
            {
                m_ep_square = from + up;
                m_hash.flip_ep(from.file());
            }
        }
    }
    if (move.type_of() == EN_PASSANT)
    {
        const auto to_ep = to - up;
        m_hash.flip_piece(Piece{~us, PAWN}, to_ep);
        remove_piece(to_ep);
    }
    if (move.type_of() == PROMOTION)
    {
        remove_piece(from);
        set_piece(move.promotion_type(), us, from);
        pc = Piece{us, move.promotion_type()};
        m_hash.promote_piece(us, pc.type(), from);
    }
    move_piece(from, to);
    m_hash.move_piece(pc, from, to);

    update();
}



inline unsigned Position::wdl_probe() const
{
    size_t ep_sq = ep_square() == NO_SQUARE ? 0 : ep_square().index() + 1;
    return tb_probe_wdl(occupancy(WHITE).value(), occupancy(BLACK).value(), occupancy(KING).value(),
                        occupancy(QUEEN).value(), occupancy(ROOK).value(), occupancy(BISHOP).value(),
                        occupancy(KNIGHT).value(), occupancy(PAWN).value(), static_cast<unsigned>(halfmove_clock()),
                        castling_rights().mask(), ep_sq, side_to_move() == WHITE);
}

inline unsigned Position::dtz_probe() const
{
    size_t ep_sq = ep_square() == NO_SQUARE ? 0 : ep_square().index() + 1;
    return tb_probe_root(occupancy(WHITE).value(), occupancy(BLACK).value(), occupancy(KING).value(),
                         occupancy(QUEEN).value(), occupancy(ROOK).value(), occupancy(BISHOP).value(),
                         occupancy(KNIGHT).value(), occupancy(PAWN).value(), static_cast<unsigned>(halfmove_clock()),
                         castling_rights().mask(), ep_sq, side_to_move() == WHITE, nullptr);
}



inline int Position::see(const Move move) const
{

    assert(move.type_of() != CASTLING);

    const Square from      = move.from_sq();
    const Square to        = move.to_sq();
    const Piece  moving_pc = piece_at(from);

    assert(moving_pc);

    const Color     us    = moving_pc.color();
    const Color     them  = ~us;
    const Direction up    = (us == WHITE) ? NORTH : SOUTH;
    const bool      is_ep = move.type_of() == EN_PASSANT;

    std::vector<int> gains;
    gains.reserve(32);

    Bitboard occ = occupancy();
    occ.unset(is_ep ? to - up : to);

    auto attackers = attacking_sq(to, occ);

    if (attackers.is_set(ksq(WHITE)) && attackers.is_set(ksq(BLACK)))
    {
        attackers.unset(ksq(WHITE));
        attackers.unset(ksq(BLACK));
    }

    auto capture = [&](const Square sq)
    {
        occ &= ~Bitboard(sq);
        attackers |= (attacks<ROOK>(to, occ) & occupancy(ROOK, QUEEN));
        attackers |= (attacks<BISHOP>(to, occ) & occupancy(BISHOP, QUEEN));
        attackers &= occ;
    };

    const Piece captured = piece_at(is_ep ? to - up : to);

    capture(from);
    gains.push_back(captured ? captured.piece_value() : 0);
    int balance = captured ? captured.piece_value() : 0;

    Color     side             = them;
    PieceType cur              = piece_type_at(from);
    bool      king_can_capture = false;

    while (true)
    {
        const Bitboard attacking = attackers & occupancy(side);
        king_can_capture         = (attackers & occupancy(~side)) == Bitboard::empty();

        if (!attacking)
            break;

        Square    chosen_sq = NO_SQUARE;
        PieceType chosen_pt = NO_PIECE_TYPE;

        for (const auto pt : PieceType::values())
        {
            const auto bb = occupancy(side, pt) & attacking;
            if (bb)
            {
                chosen_pt = pt;
                chosen_sq = Square{bb.get_lsb()};
                break;
            }
        }
        if (chosen_pt == KING && !king_can_capture)
        {
            break;
        }

        capture(chosen_sq);
        side = ~side;

        balance = -balance + cur.piece_value();
        gains.push_back(balance);

        cur = chosen_pt;
    }

    for (int i = static_cast<int>(gains.size()) - 1; i > 0; --i)
        gains[i - 1] = std::min(-gains[i], gains[i - 1]);

    return gains.empty() ? 0 : gains[0];
}

inline bool Position::is_insufficient_material() const
{
    if (occupancy(QUEEN, ROOK, PAWN) != Bitboard::empty())
        return false;

    int white_bishops = occupancy(WHITE, BISHOP).popcount();
    int black_bishops = occupancy(BLACK, BISHOP).popcount();
    int white_knights = occupancy(WHITE, KNIGHT).popcount();
    int black_knights = occupancy(BLACK, KNIGHT).popcount();


    int total_minor = white_bishops + black_bishops + white_knights + black_knights;

    if (total_minor == 0)
        return true;

    if (total_minor == 1 && (white_bishops + black_bishops) == 1)
        return true;

    if (total_minor == 1 && (white_knights + black_knights) == 1)
        return true;

    // King and Bishop vs King and Bishop (on same color)
    if (total_minor == 2 && white_bishops == 1 && black_bishops == 1)
    {
        auto get_bishop_square_color = [&](Color color) {
            const Bitboard bb = occupancy(color, BISHOP);
            if (bb) {
                Square sq = Square{bb.get_lsb()};
                return (sq.file().value() + sq.rank().value()) % 2;
            }
            return -1;
        };

        int white_color = get_bishop_square_color(WHITE);
        int black_color = get_bishop_square_color(BLACK);
        return white_color != -1 && white_color == black_color;
    }

    return false;
}


struct Positions
{
    using PosRef      = Position&;
    using ConstPosRef = const Position&;

    // Moves will not be visible through the positions span
    // They are used internally to check for repetitions
    // They do not count towards the ply limit
    explicit Positions(const Position& pos, const std::span<Move> moves = {})
    {
        m_positions.reserve(moves.size() + MAX_PLY + 1);
        m_hashes.reserve(MAX_PLY + 1);
        m_positions.emplace_back(pos);
        m_hashes.emplace_back(pos.hash(), 1);
        for (const auto m : moves)
        {
            do_move(m);
        }
        m_start_size = m_positions.size();
    }

    explicit Positions(const std::string& fen, const std::span<Move> moves = {})
    {
        m_positions.reserve(moves.size() + MAX_PLY + 1);
        m_hashes.reserve(MAX_PLY + 1);
        Position pos;
        pos.from_fen(fen);
        m_positions.emplace_back(pos);
        m_hashes.emplace_back(pos.hash(), 1);
        for (const auto m : moves)
        {
            do_move(m);
        }
        m_start_size = m_positions.size();
    }

    [[nodiscard]] std::size_t ply() const { return m_positions.size() - m_start_size; }

    std::span<Position>                     positions() { return {m_positions.data() + m_start_size - 1, ply() + 1}; }
    [[nodiscard]] std::span<const Position> positions() const { return {m_positions.data() + m_start_size - 1, ply() + 1}; }

    PosRef                    operator[](const std::size_t ply) { return positions()[ply]; }
    [[nodiscard]] ConstPosRef operator[](const std::size_t ply) const { return positions()[ply]; }

    PosRef                    operator()(const std::size_t ply) { return positions()[ply]; }
    [[nodiscard]] ConstPosRef operator()(const std::size_t ply) const { return positions()[ply]; }

    PosRef                    last() { return positions()[ply()]; }
    [[nodiscard]] ConstPosRef last() const { return positions()[ply()]; }

    void do_move(const Move move)
    {
        assert(ply() < MAX_PLY);

        m_positions.emplace_back(m_positions.back(), move);

        const auto view = m_hashes | std::views::reverse | std::views::take(last().halfmove_clock());
        const auto it   = std::ranges::find(view, last().hash(), &std::pair<hash_t, int>::first);

        int c = it != view.end() ? it->second + 1 : 1;
        m_hashes.emplace_back(last().hash(), c);
    }

    void undo_move()
    {
        assert(ply() > 0);

        m_hashes.pop_back();
        m_positions.pop_back();
    }

    [[nodiscard]] bool is_repetition() const
    {
        if (last().halfmove_clock() >= 100) return true;
        const auto view = m_hashes | std::views::reverse | std::views::take(last().halfmove_clock() + 1);
        return std::ranges::any_of(view, [&](const auto h) { return h.second >= 3; });
    }

private:
    std::vector<Position>               m_positions{};
    std::vector<std::pair<hash_t, int>> m_hashes{};
    std::size_t                         m_start_size{1};
};

#endif
