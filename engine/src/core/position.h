#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include "format.h"

#include "bitboard.h"
#include "movegen.h"
#include "types.h"
#include "zobrist.h"

#include <cstring>
#include <ostream>
#include <ranges>
#include <utility>
#include <vector>

namespace chepp {
    struct Position {
        Position() = default;
        Position(const Position& prev, const Move move) : Position(prev) {
            do_move(move);
        }
        explicit Position(const Fen& fen);
        static Position
        from_fen(const Fen& fen) {
            return Position{fen};
        }

        [[nodiscard]] zobrist::Hash
        compute_zobrist() const;
        void
        init_zobrist();

        [[nodiscard]] Square
        ep_square() const {
            return m_ep_square;
        }
        [[nodiscard]] Piece
        captured() const {
            return m_captured;
        }
        [[nodiscard]] Move
        move() const {
            return m_move;
        }
        [[nodiscard]] Piece
        moved() const {
            return m_moved;
        }
        [[nodiscard]] Color
        side_to_move() const {
            return m_color;
        }
        [[nodiscard]] uint16_t
        halfmove_clock() const {
            return m_halfmove_clock;
        }
        [[nodiscard]] uint16_t
        full_move_clock() const {
            return m_fullmove_clock;
        }
        [[nodiscard]] CastlingRights
        castling_rights() const {
            return m_crs;
        }
        [[nodiscard]] zobrist::Hash
        hash() const {
            return m_hash;
        }
        [[nodiscard]] const EnumArray<Piece, Square>&
        pieces() const {
            return m_pieces;
        }
        [[nodiscard]] Piece
        piece_at(const Square sq) const {
            return m_pieces.at(sq);
        }
        [[nodiscard]] PieceType
        piece_type_at(const Square sq) const {
            return piece_at(sq).type();
        }
        [[nodiscard]] Color
        color_at(const Square sq) const {
            return piece_at(sq).color();
        }
        [[nodiscard]] Square
        ksq(const Color c) const {
            return m_ksq.at(c);
        }

        [[nodiscard]] Bitboard
        checkers(const Color c) const {
            return m_check_mask.at(c) & occupancy(~c);
        }
        [[nodiscard]] Bitboard
        blockers(const Color c) const {
            return m_blockers.at(c);
        }
        [[nodiscard]] Bitboard
        check_mask(const Color c) const {
            return m_check_mask.at(c);
        }

        [[nodiscard]] Bitboard
        occupancy() const {
            return m_global_occupancy;
        }
        [[nodiscard]] Bitboard
        occupancy(const Color c) const {
            return m_color_occupancy.at(c);
        }
        [[nodiscard]] Bitboard
        occupancy(const PieceType p) const {
            return m_pieces_type_occupancy.at(p);
        }
        [[nodiscard]] Bitboard
        occupancy(const Color c, const PieceType p) const {
            return occupancy(p) & occupancy(c);
        }
        template <class... Ts>
        [[nodiscard]] Bitboard
        occupancy(Color c, PieceType first, const Ts... rest) const;
        template <class... Ts>
        [[nodiscard]] Bitboard
        occupancy(PieceType first, const Ts... rest) const;
        [[nodiscard]] Bitboard
        occupancy(Color c, std::initializer_list<PieceType> types) const;
        [[nodiscard]] Bitboard
        occupancy(std::initializer_list<PieceType> types) const;
        [[nodiscard]] bool
        is_occupied(const Square sq) const {
            return m_pieces.at(sq) != NO_PIECE;
        }

        [[nodiscard]] Bitboard
        attacking_sq(Square sq, Bitboard occ) const;
        [[nodiscard]] Bitboard
        attacking_sq(Square sq) const;
        [[nodiscard]] bool
        is_attacking_sq(Square sq, Color c) const;

        [[nodiscard]] bool
        is_legal(Move move) const;
        [[nodiscard]] bool
        is_capture(Move move) const;
        [[nodiscard]] Piece
        captured_by_move(Move move) const;
        void
        do_move(Move move);
        [[nodiscard]] bool
        is_quiet(Move move) const;
        [[nodiscard]] bool
        is_tactical(Move move) const;
        [[nodiscard]] bool
        is_valid(Move move) const;
        [[nodiscard]] static tl::expected<tl::monostate, std::string>
        is_ok_verbose();
        template <bool verbose = false>
        [[nodiscard]] bool
        is_ok() const;

        [[nodiscard]] bool
        in_check(const Color c) const {
            return checkers(c) != Bitboard::empty();
        }

        template <PieceType pt>
        void
        update_checkers_and_blockers(Color c);
        void
        update_checkers_and_blockers(Color c);
        void
        refresh_cache();
        void
        update();

        void
        set_piece(Piece piece, Square sq);
        void
        set_piece(PieceType piece_type, Color color, Square sq);
        void
        remove_piece(Square sq);
        void
        move_piece(Square from, Square to);

        [[nodiscard]] std::string
        to_string() const;

        [[nodiscard]] Fen
        to_fen() const;

        [[nodiscard]] unsigned
        wdl_probe() const;
        [[nodiscard]] unsigned
        dtz_probe() const;

        [[nodiscard]] int
        see(Move move) const;
        bool
        is_insufficient_material() const;

      private:
        template <Color c>
        [[nodiscard]] bool
        is_legal_for_side(Move move) const;

        // copied
        zobrist::Hash                  m_hash{};
        EnumArray<Piece, Square>       m_pieces{};
        EnumArray<Bitboard, Color>     m_color_occupancy{};
        Bitboard                       m_global_occupancy{};
        EnumArray<Bitboard, PieceType> m_pieces_type_occupancy{};
        EnumArray<Square, Color>       m_ksq{};
        CastlingRights                 m_crs{};
        Color                          m_color{};
        uint16_t                       m_halfmove_clock = 0;
        uint16_t                       m_fullmove_clock = 1;
        Square                         m_ep_square{};
        Piece                          m_captured{};
        Move                           m_move{};
        Piece                          m_moved{};
        // recomputed
        EnumArray<Bitboard, Color> m_blockers{};
        EnumArray<Bitboard, Color> m_check_mask{};
    };

    inline Position::Position(const Fen& fen)
        : m_crs(fen.crs), m_color(fen.color), m_halfmove_clock(fen.halfmove), m_fullmove_clock(fen.fullmove),
          m_ep_square(fen.ep_square) {
        for (const auto sq : Square::all()) {
            if (fen.pieces.at(sq)) {
                set_piece(fen.pieces.at(sq), sq);
            }
        }
        init_zobrist();
        refresh_cache();
        update();
    }

    inline zobrist::Hash
    Position::compute_zobrist() const {
        zobrist::Hash hash = 0;
        for (auto sq = A1; sq <= H8; sq = ++sq) {
            if (piece_at(sq) != NO_PIECE) zobrist::flip_piece(hash, piece_at(sq), sq);
        }

        if (ep_square() != NO_SQUARE) zobrist::flip_ep(hash, ep_square().file());

        if (side_to_move() == BLACK) zobrist::flip_color(hash);

        zobrist::flip_castling_rights(hash, castling_rights().mask());
        return hash;
    }

    inline void
    Position::init_zobrist() {
        m_hash = compute_zobrist();
    }

    inline Bitboard
    Position::occupancy(const Color c, const std::initializer_list<PieceType> types) const {
        return occupancy(types) & occupancy(c);
    }

    template <typename... Ts>
    [[nodiscard]] Bitboard
    Position::occupancy(const Color c, const PieceType first, const Ts... rest) const {
        return occupancy(first, rest...) & occupancy(c);
    }

    template <typename... Ts>
    Bitboard
    Position::occupancy(const PieceType first, const Ts... rest) const {
        if constexpr (sizeof...(rest) == 0)
            return occupancy(first);
        else
            return occupancy(first) | occupancy(rest...);
    }

    inline Bitboard
    Position::occupancy(const std::initializer_list<PieceType> types) const {
        Bitboard result{0};

        for (const PieceType pt : types) result |= occupancy(pt);

        return result;
    }

    inline Bitboard
    Position::attacking_sq(const Square sq, const Bitboard occ) const {
        // if a piece attacks a square sq2 from a square sq1 it would attack sq1 from sq2
        // exception made for pawns where this is true only with reversed colors
        using namespace movegen;
        return ((attacks<ROOK>(sq, occ) & occupancy(ROOK, QUEEN)) |
                (attacks<BISHOP>(sq, occ) & occupancy(BISHOP, QUEEN)) | (attacks<KNIGHT>(sq, occ) & occupancy(KNIGHT)) |
                (attacks<PAWN>(sq, occ, BLACK) & occupancy(WHITE, PAWN)) |
                (attacks<PAWN>(sq, occ, WHITE) & occupancy(BLACK, PAWN)) | (attacks<KING>(sq, occ) & occupancy(KING))) &
               occ;
    }

    inline Bitboard
    Position::attacking_sq(const Square sq) const {
        return attacking_sq(sq, occupancy());
    }

    inline bool
    Position::is_attacking_sq(const Square sq, const Color c) const {
        // if a piece attacks a square sq2 from a square sq1 it would attack sq1 from sq2
        // exception made for pawns where this is true only with reversed colors
        using namespace movegen;
        return attacks<KNIGHT>(sq, occupancy()) & occupancy(c, KNIGHT) ||
               attacks<PAWN>(sq, occupancy(), ~c) & occupancy(c, PAWN) ||
               attacks<KING>(sq, occupancy()) & occupancy(c, KING) ||
               attacks<BISHOP>(sq, occupancy()) & occupancy(c, BISHOP, QUEEN) ||
               attacks<ROOK>(sq, occupancy()) & occupancy(c, ROOK, QUEEN);
    }

    template <PieceType pt>
    void
    Position::update_checkers_and_blockers(const Color c) {
        static_assert(pt == BISHOP || pt == ROOK);
        auto enemies = occupancy(~c, pt, QUEEN) & movegen::pseudo_attack<pt>(ksq(c), ~c);
        for (const Square sq : enemies) {
            const auto line       = movegen::from_to_excl(sq, ksq(c));
            const auto on_line    = line & occupancy();
            const auto n_blockers = on_line.popcount();
            if (n_blockers == 0) {
                // check mask also contains all squares between the long range attacker and the king
                m_check_mask.at(c) |= line;
            }
            if (n_blockers == 1) {
                // blockers can be of any color
                // blocker from same color are pinned from other color can do a discovered check
                m_blockers.at(c) |= on_line;
            }
        }
    }

    inline void
    Position::update_checkers_and_blockers(const Color c) {
        m_blockers.at(c)   = bb::empty();
        m_check_mask.at(c) = attacking_sq(ksq(c)) & occupancy(~c);

        update_checkers_and_blockers<BISHOP>(c);
        update_checkers_and_blockers<ROOK>(c);
    }

    inline void
    Position::refresh_cache() {
        m_global_occupancy = occupancy(WHITE) | occupancy(BLACK);
        m_ksq              = {Square{occupancy(WHITE, KING).get_lsb()}, Square{occupancy(BLACK, KING).get_lsb()}};
    }

    inline void
    Position::update() {
        refresh_cache();
        update_checkers_and_blockers(side_to_move());
        update_checkers_and_blockers(~side_to_move());
    }

    inline void
    Position::set_piece(const Piece piece, const Square sq) {
        assert(piece);
        assert(sq);
        assert(!is_occupied(sq));
        const PieceType pt = piece.type();
        const Color     c  = piece.color();
        m_pieces_type_occupancy.at(pt) |= Bitboard(sq);
        m_color_occupancy.at(c) |= Bitboard(sq);
        m_pieces.at(sq) = piece;
    }

    inline void
    Position::set_piece(const PieceType piece_type, const Color color, const Square sq) {
        assert(piece_type);
        assert(color != NO_COLOR);
        assert(sq);
        assert(!is_occupied(sq));

        m_pieces_type_occupancy.at(piece_type) |= Bitboard(sq);
        m_color_occupancy.at(color) |= Bitboard(sq);
        m_pieces.at(sq) = Piece{color, piece_type};
    }

    inline void
    Position::remove_piece(const Square sq) {
        assert(sq);
        const Piece pc = piece_at(sq);
        m_pieces_type_occupancy.at(pc.type()) &= ~Bitboard(sq);
        m_color_occupancy.at(pc.color()) &= ~Bitboard(sq);
        m_pieces.at(sq) = NO_PIECE;
    }

    inline void
    Position::move_piece(const Square from, const Square to) {
        const Piece piece = piece_at(from);
        remove_piece(from);
        set_piece(piece, to);
    }

    inline Fen
    Position::to_fen() const {
        return {pieces(), side_to_move(), castling_rights(), ep_square(), halfmove_clock(), full_move_clock()};
    }

    inline std::string
    Position::to_string() const {
        std::string res{};
        res.reserve(400);
        auto out = std::back_inserter(res);

        fmt::format_to(out,
                       "Position\n"
                       "Side to move: {}\n"
                       "Castling rights: {}\n"
                       "EP square: {}\n"
                       "Zobrist hash: {:#X}\n",
                       side_to_move(),
                       castling_rights(),
                       ep_square(),
                       hash());

        for (auto rank = RANK_1; rank <= RANK_8; ++rank) {
            fmt::format_to(out, "{} ", (RANK_8 - rank).index() + 1);
            for (auto file = FILE_A; file <= FILE_H; ++file) {
                const Square sq{file, RANK_8 - rank};
                const Piece  pc = piece_at(sq);
                fmt::format_to(out, "{} ", pc);
            }
            fmt::format_to(out, "\n");
        }

        fmt::format_to(out,
                       "  a b c d e f g h \n"
                       "Fen: {}\n",
                       to_fen());
        return res;
    }

    template <Color c>
    bool
    Position::is_legal_for_side(const Move move) const {
        constexpr Direction down    = c == WHITE ? SOUTH : NORTH;
        const auto          from_bb = Bitboard(move.from_sq());
        const auto          to_bb   = Bitboard(move.to_sq());

        if (ksq(c) == move.from_sq()) {
            // cannot move along the ray of a long range piece
            Bitboard long_range = checkers(c) & occupancy(~c, ROOK, BISHOP, QUEEN);
            while (long_range) {
                if (const auto sq = Square{long_range.pop_lsb()};
                    movegen::are_aligned(sq, move.from_sq(), move.to_sq()) && move.to_sq() != sq) {
                    return false;
                }
            }
            // cannot move to a square that is attacked
            if (is_attacking_sq(move.to_sq(), ~c)) {
                return false;
            }
        } else if (move.type_of() == NORMAL || move.type_of() == PROMOTION) {
            // check for pins
            // if the piece is a blocker
            if (Bitboard(move.from_sq()) & blockers(c)) {
                // and if it is not moving along the line it is pinned to
                if (!(Bitboard(move.to_sq()) & movegen::line(move.from_sq(), ksq(c)))) {
                    // we are breaking the pin, move is illegal
                    return false;
                }
            }
        } else if (move.type_of() == EN_PASSANT) {
            // en passant can create discovered checks so we check for any long range attack
            // we know they have not moved. therefore we only need to update global occupancy to cast rays
            // we check if rays intersect with any long range and if they do that means a there is a check
            if (const Bitboard ep_occupancy = (occupancy() & ~from_bb | to_bb) & ~movegen::shift<down>(to_bb);
                occupancy(~c, BISHOP, QUEEN) & movegen::attacks(BISHOP, ksq(c), ep_occupancy) ||
                occupancy(~c, ROOK, QUEEN) & movegen::attacks(ROOK, ksq(c), ep_occupancy))
                return false;
        }
        return true;
    }

    inline bool
    Position::is_legal(const Move move) const {
        return side_to_move() == WHITE ? is_legal_for_side<WHITE>(move) : is_legal_for_side<BLACK>(move);
    }

    inline bool
    Position::is_capture(const Move move) const {
        const int    down           = side_to_move() == WHITE ? SOUTH : NORTH;
        const Square capture_square = move.type_of() == EN_PASSANT ? move.to_sq() + down : move.to_sq();
        return is_occupied(capture_square);
    }

    inline Piece
    Position::captured_by_move(const Move move) const {
        assert(is_capture(move));
        return move.type_of() == EN_PASSANT ? Piece{~side_to_move(), PAWN} : piece_at(move.to_sq());
    }

    inline bool
    Position::is_quiet(const Move move) const {
        return move.type_of() != PROMOTION && !is_capture(move);
    }

    inline bool
    Position::is_tactical(const Move move) const {
        return is_capture(move) || move.type_of() == PROMOTION;
    }

    inline void
    Position::do_move(const Move move) {
        zobrist::flip_color(m_hash);

        if (ep_square() != NO_SQUARE) {
            zobrist::flip_ep(m_hash, ep_square().file());
        }

        m_halfmove_clock++;
        m_fullmove_clock += m_color == BLACK;
        m_color     = ~m_color;
        m_ep_square = NO_SQUARE;
        m_captured  = NO_PIECE;
        m_move      = move;
        m_moved     = NO_PIECE;

        if (move == Move::null()) {
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
        zobrist::flip_castling_rights(m_hash, lost.mask());

        if (move.type_of() == CASTLING) {
            const auto castling_type = move.castling_type();
            assert(CastlingRights{lost}.has(castling_type));
            auto [k_from, k_to] = castling_type.king_move();
            auto [r_from, r_to] = castling_type.rook_move();

            move_piece(r_from, r_to);
            move_piece(k_from, k_to);

            zobrist::move_piece(m_hash, Piece{us, KING}, k_from, k_to);
            zobrist::move_piece(m_hash, Piece{us, ROOK}, r_from, r_to);

            update();
            return;
        }

        if (pc.type() == PAWN || is_occupied(to)) {
            m_halfmove_clock = 0;
        }

        if (move.type_of() == NORMAL || move.type_of() == PROMOTION) {

            // capture
            if (is_occupied(to)) {
                assert(color_at(to) == ~us);

                m_captured = piece_at(to);
                zobrist::flip_piece(m_hash, piece_at(to), to);
                remove_piece(to);
            }
            // set new ep square
            else if (pc.type() == PAWN && to.value() - from.value() == 2 * up) {
                // only set if ep is actually playable
                if (((movegen::pseudo_attack<PAWN>(to - up, us)) & occupancy(~us)) != bb::empty()) {
                    m_ep_square = from + up;
                    zobrist::flip_ep(m_hash, from.file());
                }
            }
        }
        if (move.type_of() == EN_PASSANT) {
            const auto to_ep = to - up;
            zobrist::flip_piece(m_hash, Piece{~us, PAWN}, to_ep);
            remove_piece(to_ep);
        }
        if (move.type_of() == PROMOTION) {
            remove_piece(from);
            set_piece(move.promotion_type(), us, from);
            pc = Piece{us, move.promotion_type()};
            zobrist::promote_piece(m_hash, us, pc.type(), from);
        }
        move_piece(from, to);
        zobrist::move_piece(m_hash, pc, from, to);

        update();
        assert(is_ok<true>());
    }

    inline int
    Position::see(const Move move) const {

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

        if (attackers.is_set(ksq(WHITE)) && attackers.is_set(ksq(BLACK))) {
            attackers.unset(ksq(WHITE));
            attackers.unset(ksq(BLACK));
        }

        auto capture = [&](const Square sq) {
            occ &= ~Bitboard(sq);
            attackers |= (movegen::attacks<ROOK>(to, occ) & occupancy(ROOK, QUEEN));
            attackers |= (movegen::attacks<BISHOP>(to, occ) & occupancy(BISHOP, QUEEN));
            attackers &= occ;
        };

        const Piece captured = piece_at(is_ep ? to - up : to);

        capture(from);
        gains.push_back(captured ? captured.piece_value() : 0);
        int balance = captured ? captured.piece_value() : 0;

        Color     side             = them;
        PieceType cur              = piece_type_at(from);
        bool      king_can_capture = false;

        while (true) {
            const Bitboard attacking = attackers & occupancy(side);
            king_can_capture         = (attackers & occupancy(~side)) == Bitboard::empty();

            if (!attacking) break;

            Square    chosen_sq = NO_SQUARE;
            PieceType chosen_pt = NO_PIECE_TYPE;

            for (const auto pt : PieceType::all()) {
                if (const auto att{occupancy(side, pt) & attacking}) {
                    chosen_pt = pt;
                    chosen_sq = Square{att.get_lsb()};
                    break;
                }
            }
            if (chosen_pt == KING && !king_can_capture) {
                break;
            }

            capture(chosen_sq);
            side = ~side;

            balance = -balance + cur.piece_value();
            gains.push_back(balance);

            cur = chosen_pt;
        }

        for (size_t i = 0; i < gains.size() - 1; ++i) {
            const auto idx = gains.size() - 1 - i;
            gains[idx - 1] = std::min(-gains[idx], gains[idx - 1]);
        }

        return gains.empty() ? 0 : gains[0];
    }

    inline bool
    Position::is_insufficient_material() const {
        if (occupancy(QUEEN, ROOK, PAWN) != Bitboard::empty()) {
            return false;
        }

        const int white_bishops{occupancy(WHITE, BISHOP).popcount()};
        const int black_bishops{occupancy(BLACK, BISHOP).popcount()};
        const int white_knights{occupancy(WHITE, KNIGHT).popcount()};
        const int black_knights{occupancy(BLACK, KNIGHT).popcount()};

        const int total_minor{white_bishops + black_bishops + white_knights + black_knights};

        if (total_minor == 0) {
            return true;
        }
        if (total_minor == 1 && (white_bishops + black_bishops) == 1) {
            return true;
        }
        if (total_minor == 1 && (white_knights + black_knights) == 1) {
            return true;
        }

        // King and Bishop vs King and Bishop (on same color)
        if (total_minor == 2 && white_bishops == 1 && black_bishops == 1) {
            auto get_bishop_square_color = [&](const Color color) {
                if (const Bitboard occ{occupancy(color, BISHOP)}) {
                    const Square sq{occ.get_lsb()};
                    return (sq.file().value() + sq.rank().value()) % 2;
                }
                return -1;
            };

            const int white_color{get_bishop_square_color(WHITE)};
            const int black_color{get_bishop_square_color(BLACK)};
            return white_color != -1 && white_color == black_color;
        }

        return false;
    }

    inline void
    make_all_promotions(MoveList& list, const Square from, const Square to) {
        for (const auto pt : {QUEEN, ROOK, KNIGHT, BISHOP}) list.push_back(Move::make<PROMOTION>(from, to, pt));
    }

    template <move_type_t T>
    void
    add_moves_from_bb(MoveList& list, const Bitboard b, const int delta) {
        b.for_each_square([&](const Square to) { list.push_back(Move::make<T>(to - delta, to)); });
    }

    inline void
    add_promotions(MoveList& list, const Bitboard b, const int delta) {
        b.for_each_square([&](const Square to) { make_all_promotions(list, to - delta, to); });
    }

    template <Color c>
    void
    gen_pawn_moves(const Position& pos, MoveList& list) {
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
            Bitboard single_push = movegen::shift<up>(pawns & ~bb_promotion_rank) & available;
            Bitboard double_push = movegen::shift<up>(single_push & bb_third_rank) & available & check_mask;
            single_push &= check_mask;

            add_moves_from_bb<NORMAL>(list, single_push, up);
            add_moves_from_bb<NORMAL>(list, double_push, up + up);
        }
        // promotion
        if (const Bitboard promotions = pawns & bb_promotion_rank) {
            Bitboard push       = movegen::shift<up>(promotions) & available & check_mask;
            Bitboard take_right = movegen::shift<up_right>(promotions) & enemy & check_mask;
            Bitboard take_left  = movegen::shift<up_left>(promotions) & enemy & check_mask;

            add_promotions(list, push, up);
            add_promotions(list, take_right, up_right);
            add_promotions(list, take_left, up_left);
        }
        // capture
        {

            auto handle_capture = [&pos, &list](const Bitboard b, const int delta) {
                for (const Square to : b) {
                    if (to == pos.ep_square())
                        list.push_back(Move::make<EN_PASSANT>(to - delta, to));
                    else
                        list.push_back(Move::make<NORMAL>(to - delta, to));
                }
            };
            const Bitboard ep_capture_mask   = (check_mask & movegen::shift<down>(ep_bb)) ? ep_bb : bb::empty();
            const Bitboard possible_captures = (enemy | ep_bb) & (check_mask | ep_capture_mask);
            handle_capture(movegen::shift<up_right>(pawns & ~bb_promotion_rank) & possible_captures, up_right);
            handle_capture(movegen::shift<up_left>(pawns & ~bb_promotion_rank) & possible_captures, up_left);
        }
    }

    template <PieceType pc>
    void
    gen_pc_moves(const Position& pos, MoveList& list) {
        const Color    c = pos.side_to_move();
        const Bitboard check_mask{pos.check_mask(c) == bb::empty() ? bb::full() : pos.check_mask(c)};

        for (const Square from : pos.occupancy(c, pc)) {
            for (const Square to : movegen::attacks<pc>(from, pos.occupancy()) & ~pos.occupancy(c) & check_mask) {
                list.push_back(Move::make<NORMAL>(from, to));
            }
        }
    }

    inline void
    gen_king_moves(const Position& pos, MoveList& list) {
        const Color    c     = pos.side_to_move();
        const Square   from  = pos.ksq(c);
        const Bitboard moves = movegen::attacks<KING>(from, pos.occupancy());

        for (const auto to : moves & (~pos.occupancy() | pos.occupancy(~c)))
            list.push_back(Move::make<NORMAL>(from, to));
        ;

        const CastlingRights rights = pos.castling_rights();

        if (pos.check_mask(c) || !rights.has_any_color(c)) return;

        for (const auto side : {KINGSIDE, QUEENSIDE}) {
            if (const auto type = CastlingType{c, side}; rights.has(type)) {
                auto [k_from, k_to] = type.king_move();
                auto [r_from, r_to] = type.rook_move();
                assert(pos.piece_at(k_from) == Piece(c, KING));
                bool            safe = (movegen ::from_to_excl(k_from, r_from) & pos.occupancy()) == bb::empty();
                const Direction dir  = direction_from(k_from, k_to);
                assert(dir != NO_DIRECTION);
                for (auto sq = k_from + dir; sq != k_to && safe; sq = sq + dir) {
                    safe &= !pos.is_attacking_sq(sq, ~c);
                }
                if (safe) list.push_back(Move::make<CASTLING>(k_from, k_to, type));
            }
        }
    }

    template <Color c>
    MoveList
    gen_moves_for_side(const Position& pos) {
        MoveList  list;
        const int n_checkers = pos.checkers(c).popcount();
        assert(n_checkers <= 2);

        if (n_checkers != 2) {
            gen_pawn_moves<c>(pos, list);
            gen_pc_moves<BISHOP>(pos, list);
            gen_pc_moves<KNIGHT>(pos, list);
            gen_pc_moves<ROOK>(pos, list);
            gen_pc_moves<QUEEN>(pos, list);
        }
        gen_king_moves(pos, list);
        return list;
    }

    inline MoveList
    gen_moves(const Position& pos) {
        if (pos.side_to_move() == WHITE) return gen_moves_for_side<WHITE>(pos);
        return gen_moves_for_side<BLACK>(pos);
    }

    inline MoveList
    gen_legal(const Position& pos) {
        MoveList legal;
        std::ranges::copy_if(gen_moves(pos), std::back_inserter(legal), std::bind_front(&Position::is_legal, pos));
        return legal;
    }

    inline void
    perft(const Position& prev, const int ply, size_t& out) {
        MoveList l = gen_legal(prev);

        if (ply == 1) {
            out += l.size();
            return;
        }

        for (const auto move : l) {
            Position next{prev};
            next.do_move(move);
            perft(next, ply - 1, out);
        }
    }

    inline void
    perft_divide(const Position& prev, int depth) {
        MoveList l     = gen_moves(prev);
        size_t   total = 0;

        for (auto mv : l) {
            Position next = prev;
            next.do_move(mv);

            size_t nodes = 0;
            perft(next, depth - 1, nodes);

            fmt::println(stdout, "{} {} {}: {}", prev.piece_at(mv.from_sq()), mv.from_sq(), mv.to_sq(), nodes);

            total += nodes;
        }
        fmt::println(stdout, "Total {}", total);
    }

    // Expensive function, should only be called for user input validation
    inline bool
    Position::is_valid(const Move move) const {
        return std::ranges::contains(gen_legal(*this), move);
    }

    inline tl::expected<tl::monostate, std::string>
    Position::is_ok_verbose() {
        /**
        auto err = [](std::string msg) -> std::expected<tl::monostate, std::string> {
            return std::unexpected(std::move(msg));
        };

        if (m_color == NO_COLOR) return err("Invalid color");

        EnumArray<Bitboard, Color>     color_occ_local{};
        EnumArray<Bitboard, PieceType> type_occ_local{};
        Bitboard                       global_local = bb::empty();

        for (auto sq : Square::all()) {
            const Piece pc = m_pieces.at(sq);
            if (pc) {
                const PieceType pt = pc.type();
                const Color     c  = pc.color();

                if (c == NO_COLOR || !pt) return err("Invalid piece at " + std::string(sq.to_string()));

                type_occ_local.at(pt).set(sq);
                color_occ_local.at(c).set(sq);
                global_local.set(sq);
            }
        }

        if (global_local != m_global_occupancy) return err("occupancy is incoherent with pieces");

        for (const auto c : Color::all()) {
            if (color_occ_local.at(c) != m_color_occupancy.at(c))
                return err(std::string("colore occupancy is incoherent for side") + std::string(c.to_string()));
        }

        for (const auto pt : PieceType::all()) {
            if (type_occ_local.at(pt) != m_pieces_type_occupancy.at(pt))
                return err(std::string("piece type occupancy is incoherent for type ") + std::string(pt.to_string()));
        }

        for (const auto c : Color::all()) {
            const Bitboard king_bb    = type_occ_local.at(KING) & color_occ_local.at(c);
            const int      king_count = king_bb.popcount();
            if (king_count != 1)
                return err(std::string("Nb of kings for ") + std::string(c.to_string()) + " is " +
                           std::to_string(king_count));

            const Square king_sq = Square{king_bb.get_lsb()};
            if (m_ksq.at(c) != king_sq)
                return err(std::string("King square incoherence for side ") + std::string(c.to_string()));

            if (piece_at(king_sq) != Piece{c, KING}) return err("Piece at ksq is not KING");
        }

        for (const auto side : {KINGSIDE, QUEENSIDE}) {
            for (const auto c : Color::all()) {
                if (const CastlingType ct{c, side}; m_crs.has(ct)) {
                    auto [k_from, k_to] = ct.king_move();
                    auto [r_from, r_to] = ct.rook_move();
                    if (piece_at(k_from) != Piece{c, KING})
                        return err("Expected castling rights but king has moved (" + std::string(k_from.to_string()) +
                                   ").");
                    if (piece_at(r_from) != Piece{c, ROOK})
                        return err("Expected castling rights but rook has moved (" + std::string(r_from.to_string()) +
                                   ").");
                }
            }
        }

        if (m_ep_square) {
            const Rank r = m_ep_square.rank();
            if (!(r == RANK_3 || r == RANK_6)) return err("En passant square must be rank 3 or 6.");

            if (r == RANK_6) {
                const Square pawn_sq = m_ep_square + SOUTH;
                if (pawn_sq == NO_SQUARE) return err("Invalid ep square");
                if (piece_at(pawn_sq) != Piece{BLACK, PAWN})
                    return err("Invalid ep square, must have a black pawn on rank 5");
                if (side_to_move() != WHITE) return err("Invalid ep square, must be white to move");
            } else {
                const Square pawn_sq = m_ep_square + NORTH;
                if (!pawn_sq) return err("Invalid ep square");
                if (piece_at(pawn_sq) != Piece{WHITE, PAWN})
                    return err("Invalid ep square, must have a wite pawn on rank 4");
                if (side_to_move() != BLACK) return err("Invalid ep square, must be black to move");
            }
        }

        if (compute_zobrist() != m_hash) return err("Incoherent zobrist hash.");

        if (m_fullmove_clock < 1) return err("fullmove clock must be >= 1");

        if (m_move == Move::null()) {
            if (!m_moved || !m_captured) return err("m_move == null but recorded a piece movement / capture");
        } else {
            if (!m_moved && (m_captured || m_moved)) return err("m_move is not null but did not record any piece
        movement");
        }

        if (in_check(~side_to_move())) {
            return err("Enemy king can be captured");
        }

        **/
        return tl::expected<tl::monostate, std::string>(tl::monostate{});
    }

    template <bool verbose>
    bool
    Position::is_ok() const {
        auto ok = is_ok_verbose();
        if (!ok && verbose) {
            fmt::println(stderr, "{}", ok.error());
        }
        return ok.has_value();
    }

    struct Positions {
        using Ref      = Position&;
        using ConstRef = const Position&;
        using Handle   = VectorHandle<Position>;

        void
        clear() {
            m_positions.clear();
            m_hashes.clear();
            m_start_size = 0;
        }

        template <bool validate = false>
        tl::expected<tl::monostate, std::string>
        set_pos(const Position& pos, const std::span<const Move> moves = {}) {
            clear();

            if constexpr (validate) {
                auto ok = pos.is_ok_verbose();
                if (!ok) {
                    return tl::unexpected(ok.error());
                }
            }

            m_positions.reserve(moves.size() + MAX_PLY + 1);
            m_hashes.reserve(MAX_PLY + 1);

            m_positions.emplace_back(pos);
            m_hashes.emplace_back(pos.hash(), 1);

            for (std::size_t i = 0; i < moves.size(); ++i) {
                const Move m = moves[i];

                if constexpr (validate) {
                    if (!m_positions.back().is_valid(m)) {
                        return tl::unexpected(fmt::format("Invalid move in moves[{}]: {} (at ply {})", i, m, i + 1));
                    }
                }

                do_move(m);
            }

            m_start_size = m_positions.size();
            return tl::expected<tl::monostate, std::string>(tl::monostate{});
        }

        template <bool validate = false>
        tl::expected<tl::monostate, std::string>
        set_fen(const std::string& fen_string, const std::span<Move> moves = {}) {
            auto fen = Fen::from_string(fen_string);
            if (!fen) {
                return tl::unexpected(fen.error());
            }
            const Position pos{*fen};

            return set_pos<validate>(pos, moves);
        }

        [[nodiscard]] uint32_t
        ply() const {
            return static_cast<uint32_t>(m_positions.size() - m_start_size);
        }

        std::span<Position>
        positions() {
            return {m_positions.data() + m_start_size - 1, m_positions.size() - m_start_size + 1};
        }
        [[nodiscard]] std::span<const Position>
        positions() const {
            return {m_positions.data() + m_start_size - 1, m_positions.size() - m_start_size + 1};
        }

        Ref
        operator[](const std::size_t ply) {
            return positions()[ply];
        }
        [[nodiscard]] ConstRef
        operator[](const std::size_t ply) const {
            return positions()[ply];
        }

        Ref
        operator()(const std::size_t ply) {
            return positions()[ply];
        }
        [[nodiscard]] ConstRef
        operator()(const std::size_t ply) const {
            return positions()[ply];
        }

        Ref
        last() {
            return positions()[ply()];
        }
        [[nodiscard]] ConstRef
        last() const {
            return positions()[ply()];
        }

        void
        do_move(const Move move) {
            assert(ply() < MAX_PLY);

            m_positions.emplace_back(m_positions.back(), move);

            const auto view = m_hashes | std::views::reverse | std::views::take(last().halfmove_clock());
            const auto it   = std::ranges::find(view, last().hash(), &std::pair<zobrist::Hash, int>::first);

            int c = it != view.end() ? it->second + 1 : 1;
            m_hashes.emplace_back(last().hash(), c);
        }

        void
        undo_move() {
            assert(ply() > 0);

            m_hashes.pop_back();
            m_positions.pop_back();
        }

        [[nodiscard]] bool
        is_repetition() const {
            const auto view = m_hashes | std::views::reverse | std::views::take(last().halfmove_clock() + 1);
            return std::ranges::any_of(view, [&](const auto h) { return h.second >= 3; });
        }

        [[nodiscard]] bool
        is_50_move_rule() const {
            return m_positions.back().halfmove_clock() >= 100;
        }

        Handle
        handle_to_last() {
            return VectorHandle{&m_positions, static_cast<unsigned>(m_positions.size() - 1)};
        }

      private:
        std::vector<Position>                      m_positions{};
        std::vector<std::pair<zobrist::Hash, int>> m_hashes{};
        std::size_t                                m_start_size{0};
    };
} // namespace chepp

#endif
