#ifndef SEARCHER_H
#define SEARCHER_H

#include "core.h"
#include "history.h"
#include "move_ordering.h"
#include "network.h"
#include "search_stack.h"
#include "tm.h"
#include "tt.h"

#include <array>
#include <chrono>
#include <functional>
#include <latch>
#include <memory>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chepp {
    inline std::function<int(bool, int, int)> default_lmr = [](const bool quiet, const int d, const int m) {
        return quiet ? static_cast<int>(0.2 + std::log(m) * std::log(d) / 3.55)
                     : static_cast<int>(1.35 + std::log(m) * std::log(d) / 2.75);
    };

    inline std::function<int(bool, int)> default_lmp = [](const bool improving, int d) {
        d = std::clamp(d, 0, 8);
        return improving ? static_cast<int>(4 + 4 * d * d / 4.5) : static_cast<int>(2.5 + 2 * d * d / 4.5);
    };

    struct SearchThread {
        // can be overridden by UCI default params, and themself overridden by user
        struct Parameters {
            int  n_pv{1};
            bool use_syzygy{false};

            int aspiration_window_activation_depth{8};
            int aspiration_window_default_value{50};
            int aspiration_window_multiplicative_factor{2};

            std::function<int(bool, int)>      lmp{default_lmp};
            std::function<int(bool, int, int)> lmr{default_lmr};

            MoveSelector::Params scoring_parameters{};

            int max_history         = 16384;
            int max_counter_history = 16384;

            int bonus_mul_div = 500;
            int bonus_pow_div = 500;
            int bonus_rise = 10;
            int bonus_cutoff = 10;
            int bonus_depth_mul = 1;
            int bonus_tt = 10;
            int malus_mul_div = 500;
            int malus_depth_mul = 1;


            int tt_replacement_threshold{3};
        };

        struct Cache {
            std::array<std::array<int, MAX_MOVES>, 2>                      lmp{};
            std::array<std::array<std::array<int, MAX_MOVES>, MAX_PLY>, 2> lmr{};
        };

        struct Statistics {
            uint64_t                                       nodes{0};
            uint64_t                                       tt_hits{0};
            uint64_t                                       tb_hits{0};
            std::chrono::high_resolution_clock::time_point t_start{};
        };

        struct PvLine {
            int32_t  score{0};
            MoveList line{};
        };

        using PvLines = std::vector<PvLine>;

        explicit SearchThread(const Parameters&                           parameters,
                              const int                                   id,
                              TimeManager*                                tm,
                              TT*                                         tt,
                              const Positions&                            pos,
                              const std::shared_ptr<nnue::Arch::Network>& network)
            : m_thread_id(id), m_parameters(parameters), m_base_pos(pos), m_tm(tm), m_tt(tt), m_search_stack(pos, network) {
            init_cache();
        }
        // control
        int        m_thread_id;
        Parameters m_parameters;
        Positions m_base_pos;
        // shared states
        TimeManager* m_tm;
        TT*          m_tt;
        // thread local states
        Cache       m_cache;
        Statistics  m_statistics;
        SearchStack m_search_stack;
        Move        root_best_move;

        void
        init_cache() {
            for (int improving = 0; improving < 2; ++improving) {
                for (int d = 1; d < MAX_MOVES; ++d) {
                    m_cache.lmp[improving][d] = m_parameters.lmp(improving, d);
                }
            }

            for (int quiet = 0; quiet < 2; ++quiet) {
                for (int d = 1; d < MAX_PLY; ++d) {
                    for (int m = 1; m < MAX_MOVES; ++m) {
                        m_cache.lmr[quiet][d][m] = m_parameters.lmr(quiet, d, m);
                    }
                }
            }
        }

        [[nodiscard]] int
        ply() const {
            return m_search_stack.ply();
        }
        [[nodiscard]] SearchStack::Node&
        ss() {
            return m_search_stack[m_search_stack.ply()];
        }
        [[nodiscard]] const SearchStack::Node&
        ss() const {
            return m_search_stack[m_search_stack.ply()];
        }
        void
        do_move(const Move move, const bool update_nnue = true) {
            m_search_stack.do_move(move, update_nnue);
        }
        void
        undo_move(const bool update_nnue = true) {
            m_search_stack.undo_move(update_nnue);
        }

        int32_t
        evaluate() {
            // assert(!ss().position->in_check(ss().position->side_to_move()));
            // assert(!is_draw());

            auto eval = ss().network->forward(ss().position->side_to_move());
            eval      = std::clamp(eval, LOSS_TB + 1, WIN_TB - 1);
            eval -= eval * ss().position->halfmove_clock() / 101;
            return eval;
        }

        [[nodiscard]] bool
        is_draw() const {
            return ss().is_repetition || ss().position->halfmove_clock() >= 100 ||
                   ss().position->is_insufficient_material();
        }

        [[nodiscard]] std::vector<Move>
        get_pv(Positions positions, const Move first_move) const {
            Move     move = first_move;
            std::vector<Move> moves{};
            while (true) {
                if (!positions.last().is_valid(move)) {
                    break;
                }
                if (positions.ply() >= MAX_PLY) {
                    break;
                }
                moves.push_back(move);
                positions.do_move(move);
                if (positions.is_repetition() || positions.last().halfmove_clock() >= 100) {
                    positions.undo_move();
                    break;
                }
                auto tt_hit = m_tt->probe(positions.last().hash());
                if (!tt_hit) {
                    break;
                }
                if (moves.size() == MAX_MOVES) {
                    break;
                }
                move = tt_hit->move;
            }
            return moves;
        }

        static std::string
        format_pv_line(const std::vector<Move>& pv_line) {
            std::string out;
            for (const auto m : pv_line) {
                fmt::format_to(std::back_inserter(out), "{} ", m);
            }
            return out;
        }

        [[nodiscard]] bool
        causes_draw(const Move move) {
            bool ret = false;
            do_move(move, false);
            if (is_draw()) ret = true;
            undo_move(false);
            return ret;
        }

        void
        IterativeDeepening();
        int
        AspirationWindow(int depth, int prev_eval);
        int
        Negamax(int depth, int alpha, int beta, bool cutnode);
        int
        QSearch(int alpha, int beta);
    };

    inline void
    SearchThread::IterativeDeepening() {
        int prev_eval        = evaluate();
        m_statistics.t_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < m_parameters.n_pv; ++i) {
            for (int depth = 1; m_tm->update_depth(depth), !m_tm->should_stop(); ++depth) {
                const int eval = AspirationWindow(depth, prev_eval);
                assert(ss().position);

                assert(eval > -INF && eval < INF);

                if (!m_tm->should_stop()) // aspiration window got cancelled, we discard the result
                {
                    prev_eval = eval;

                    if (m_thread_id == 0) {
                        std::string score;
                        if (eval >= MATE_IN_MAX_PLY) {
                            score.append("mate ");
                            score.append(std::to_string((MATE - eval) / 2));
                        } else if (eval <= MATED_IN_MAX_PLY) {
                            score.append("mate ");
                            score.append(std::to_string((MATED - eval) / 2));
                        } else {
                            score.append("cp ");
                            score.append(std::to_string(eval));
                        }

                        auto t_now = std::chrono::high_resolution_clock::now();
                        auto time_since_start =
                            std::chrono::duration_cast<std::chrono::milliseconds>(t_now - m_statistics.t_start);
                        time_since_start = std::max(time_since_start, std::chrono::milliseconds(1));
                        auto nps         = m_statistics.nodes / time_since_start.count() * 1000;
                        auto pv          = get_pv(m_base_pos, root_best_move);
                        fmt::println(stdout,
                                     "info score {} depth {} nodes {} nps {} tb_hits {} pv {}",
                                     score,
                                     depth,
                                     m_statistics.nodes,
                                     nps,
                                     m_statistics.tb_hits,
                                     format_pv_line(pv));
                        std::fflush(stdout);

                        TimeManager::UpdateInfo update_info;
                        update_info.eval = eval;
                        m_tm->adjust_time(update_info);
                    }
                }
            }
        }
    }

    inline int
    SearchThread::AspirationWindow(const int depth, const int prev_eval) {
        if (depth < m_parameters.aspiration_window_activation_depth) {
            int eval       = Negamax(depth, -INF_SCORE, INF_SCORE, false);
            root_best_move = ss().best_move;
            return eval;
        }

        int window = m_parameters.aspiration_window_default_value;
        int alpha  = prev_eval - window;
        int beta   = prev_eval + window;

        int eval = Negamax(depth, alpha, beta, false);

        if (m_tm->should_stop()) {
            return eval;
        }

        while (eval <= alpha || eval >= beta) {
            if (eval <= alpha) {
                window = window * m_parameters.aspiration_window_multiplicative_factor;
                alpha  = std::clamp(eval - window, -INF_SCORE, INF_SCORE);
            } else if (eval >= beta) {
                window         = window * m_parameters.aspiration_window_multiplicative_factor;
                beta           = std::clamp(eval + window, -INF_SCORE, INF_SCORE);
                root_best_move = ss().best_move;
            }

            eval = Negamax(depth, alpha, beta, false);
            if (m_tm->should_stop()) {
                return eval;
            }
        }

        root_best_move = ss().best_move;
        return eval;
    }

    inline int
    SearchThread::Negamax(int depth, int alpha, int beta, bool cutnode) {
        // assert(depth >= 0);

        // quiescence search supposed to prevent horizon effect
        if (depth <= 0) return QSearch(alpha, beta);

        if (m_thread_id == 0 && m_statistics.nodes % std::min((bit::get_msb(m_statistics.nodes) * bit::get_msb(m_statistics.nodes) * 10), 4096) == 0) {
            m_tm->update_time();
        }

        const int         alpha_org = alpha;
        const bool        is_pv     = beta - alpha > 1;
        const bool        is_root   = ply() == 0;
        const bool        in_check  = ss().position->in_check(ss().position->side_to_move());
        Positions::Handle pos       = ss().position;
        bool              improving = false;
        int               eval      = 0;

        // increase depth if we are in check
        if (in_check) {
            depth++;
        }

        m_statistics.nodes++;

        if (!is_root) {
            if (ply() >= MAX_PLY - 1) {
                return evaluate();
            }
            if (is_draw()) {
                return 0;
            }
            // this speeds up mate cases
            // our worse move is to be mated on the spot
            alpha = std::max(alpha, mated_in(ply()));
            // their best move is to mate next turn
            beta = std::min(beta, mate_in(ply() + 1));
            if (alpha >= beta) {
                return alpha;
            }
        }

        MoveList moves;

        // Probe the TT to see if we have a candidate score
        auto tt_hit  = m_tt->probe(ss().position->hash());
        Move tt_move = tt_hit ? tt_hit->move : Move::none();
        if (!tt_move) tt_hit = std::nullopt;
        if (tt_hit) {
            moves = gen_moves(ss().position());
            if (auto it = ranges::find(moves, tt_move); it == moves.end()) {
                tt_hit = std::nullopt;
            }
        }
        if (tt_hit) {
            if (!ss().position->is_legal(tt_move)) {
                tt_hit = std::nullopt;
            }
        }
        if (tt_hit) tt_hit = causes_draw(tt_move) ? std::nullopt : tt_hit;
        int tt_score = tt_hit ? TT::read_score(tt_hit->score, ply()) : 0;
        if (!is_pv && tt_hit && !ss().excluded && tt_hit->depth >= depth) {
            if (tt_hit->bound == TT::Bound::EXACT || (tt_hit->bound == TT::Bound::LOWER && tt_score >= alpha) ||
                (tt_hit->bound == TT::Bound::UPPER && tt_score <= beta)) {
                m_statistics.tt_hits++;
                return tt_score;
            }
        }
        ss().static_eval = eval = tt_hit ? tt_hit->static_eval : evaluate();
        if (ss().excluded) tt_hit = std::nullopt;

        improving = !in_check;
        if (ply() > 1) improving &= ss().static_eval > ss().prev->prev->static_eval;

        if (in_check || ss().excluded) {
            ss().static_eval = eval = 0;
            improving               = false;
        }

        if (!is_pv && !in_check && !is_root && !ss().excluded) {
            if (tt_hit) {
                eval = tt_score;
            }

            if (ply() >= 2 && depth < 9 && eval >= beta &&
                pos->occupancy().popcount() > 4 // disable when nnue uses last bucket with high evals
                && eval - ((depth - improving) * 77) - ss().prev->static_eval / 400 >= beta &&
                std::abs(eval) < MATED_IN_MAX_PLY && (!tt_move)) {
                return eval;
            }

            if (cutnode && ss().static_eval > (beta - 76 * improving) &&
                pos->occupancy(KNIGHT, BISHOP, ROOK, QUEEN).popcount() >= 1 && depth >= 3 &&
                (!tt_hit || tt_hit->bound == TT::LOWER || eval >= beta)) {
                assert(pos->move().is_ok());

                int reduction = 3 + depth / 3;

                do_move(Move::null());
                int score = -Negamax(depth - reduction, -beta, -beta + 1, !cutnode);
                undo_move();

                if (m_tm->should_stop()) {
                    return 0;
                }

                if (score >= beta && score < MATE_IN_MAX_PLY) {
                    return score;
                }
            }

            int rbeta = std::min(beta + 100, MATE - MAX_PLY - 1);
            if (depth >= 3 && abs(beta) < MATE_IN_MAX_PLY && (!tt_hit || eval >= rbeta || tt_hit->depth < depth - 3)) {
                if (moves.empty()) moves = gen_moves(pos());
                MoveSelector move_selector{moves | std::views::filter(std::bind_front(&Position::is_tactical, pos())) |
                                               std::views::filter(std::bind_front(&Position::is_legal, pos())),
                                           MoveSelector::Stage::ProbCut,
                                           ss(),
                                           m_parameters.scoring_parameters,
                                           tt_move};

                int score = 0;

                for (const auto [m, s] : move_selector) {
                    if (s < move_selector.params().good_capture) continue;
                    if (m == ss().excluded) continue;

                    do_move(m);
                    score = -QSearch(-rbeta, -rbeta + 1);

                    if (score >= rbeta) {
                        score = -Negamax(depth - 4, -rbeta, -rbeta + 1, !cutnode);
                    }

                    undo_move();

                    if (score >= rbeta) {
                        m_tt->store(ss().position->hash(),
                                    static_cast<uint16_t>(depth - 3),
                                    static_cast<int16_t>(score),
                                    TT::LOWER,
                                    m,
                                    static_cast<int16_t>(ss().static_eval));
                        return score;
                    }
                }
            }

            if (false && eval - 63 + 182 * depth <= alpha) {
                return QSearch(alpha, beta);
            }
        }

        if (cutnode && depth >= 7 && tt_move == Move::none()) {
            depth--;
        }

        int  best_score      = -INF;
        int  move_count      = 0;
        int  score           = -INF;
        Move local_best_move = Move::none();

        bool skip_quiets = false;

        if (moves.empty()) moves = gen_moves(pos());
        MoveSelector::Stage stage =
            false && is_root && depth > 7 ? MoveSelector::Stage::Root : MoveSelector::Stage::Search;
        MoveSelector selector{moves | std::views::filter(std::bind_front(&Position::is_legal, pos())),
                              stage,
                              ss(),
                              m_parameters.scoring_parameters,
                              tt_move};

        struct ExploredMove {
            Move move;
            bool alpha_raise;
            bool beta_cutoff;
            int  score;
        };

        ArrayStack<ExploredMove, MoveList::capacity()> explored_quiets;
        ArrayStack<ExploredMove, MoveList::capacity()> explored_tacticals;

        for (const auto [m, s] : selector) {
            if (m == ss().excluded) continue;

            bool  is_quiet  = ss().position->is_quiet(m);
            auto& explored  = is_quiet ? explored_quiets : explored_tacticals;
            int   extension = 0;

            bool refutation_move = (ss().killer1 == m || ss().killer2 == m);

            if (is_quiet && skip_quiets) continue;

            if (!is_root && best_score > MATED) {

                int lmr_depth =
                    m_cache.lmr.at(is_quiet).at(std::min(depth, MAX_PLY - 1)).at(std::min(move_count, MAX_MOVES - 1));

                if (is_quiet) {
                    if (!in_check && !is_pv && depth <= 7 &&
                        (int)explored_quiets.size() > m_cache.lmp.at(improving).at(depth)) {
                        skip_quiets = true;
                        continue;
                    }

                    if (lmr_depth < 3 && !refutation_move && s < -4000 * depth) {
                        continue;
                    }

                    if (false && lmr_depth <= 6 && !in_check && eval + 217 + 71 * depth <= alpha) {
                        skip_quiets = true;
                    }

                } else {
                    if (depth <= 6 && ss().position->see(m) < -15 * depth * depth) {
                        continue;
                    }
                }
            }

            if (!is_root && tt_hit && depth >= (6 + is_pv) && (m == tt_move) && tt_hit->bound == TT::LOWER &&
                abs(tt_score) < MATE && tt_hit->depth >= depth - 3 && !ss().excluded) {
                int singular_beta  = tt_score - depth;
                int singular_depth = (depth - 1) / 2;

                ss().excluded      = tt_move;
                int singular_score = Negamax(singular_depth, singular_beta - 1, singular_beta, cutnode);
                ss().excluded      = Move::none();

                if (singular_score < singular_beta && ss().single_extensions < 5) {
                    extension = 1;
                    ss().single_extensions++;
                    if (!is_pv && singular_score < singular_beta - 20 && ss().double_extensions < 2) {
                        extension = 2;
                        ss().double_extensions++;
                    }
                } else if (singular_beta >= beta) {
                    return singular_beta;
                } else if (tt_score >= beta) {
                    extension = -2;
                } else if (tt_score <= singular_score || cutnode) {
                    extension = -1;
                }
            }

            int new_depth = depth + extension;

            auto start = m_statistics.nodes;

            do_move(m);
            m_tt->prefetch(ss().position->hash());

            m_statistics.nodes++;
            move_count++;

            if (is_root && depth == 1 && move_count == 1) {
                ss().best_move = m;
            }

            explored.push_back({m, false, false, 0});

            bool do_full_search = !is_root || move_count > 1;

            if (!in_check && do_full_search && depth >= 3 && move_count > (2 + 2 * is_pv)) {
                int reduction =
                    m_cache.lmr.at(is_quiet).at(std::min(depth, MAX_PLY - 1)).at(std::min(move_count, MAX_MOVES - 1));

                reduction += !improving;
                reduction += is_pv;
                reduction += is_quiet;

                reduction -= is_quiet ? refutation_move ? 2 : s / 4000 : 0;

                reduction = std::min(depth - 1, std::max(1, reduction));

                score = -Negamax(new_depth - reduction, -alpha - 1, -alpha, true);

                do_full_search = score > alpha && reduction != 1;

                bool deeper = score > best_score + 70 + 12 * (new_depth - reduction);
                new_depth += deeper;
            }

            if (do_full_search) {
                score = -Negamax(new_depth - 1, -alpha - 1, -alpha, !cutnode);
            }

            if (is_pv && (move_count == 1 || (score > alpha && score < beta))) {
                score = -Negamax(new_depth - 1, -beta, -alpha, false);
            }

            undo_move();

            auto end = m_statistics.nodes;

            if (is_root) {
                if (!ss().refutation_history->contains(m)) ss().refutation_history->emplace(m, 0);
                ss().refutation_history->at(m) += end - start;
            }

            if (m_tm->should_stop() && !is_root) {
                return 0;
            }

            if (score > best_score) {
                explored.back().score = score;
                best_score            = score;
                local_best_move       = m;
                if (score > alpha) {
                    explored.back().alpha_raise = true;
                    alpha                       = score;

                    if (alpha >= beta) {
                        explored.back().beta_cutoff = true;
                        if (is_quiet) {
                            if (ss().killer1 != m) {
                                ss().killer2 = ss().killer1;
                                ss().killer1 = m;
                            }
                        }
                        break;
                    }
                }
            }

            if (m_tm->should_stop() && is_root && local_best_move) {
                break;
            }
        }

        if (move_count == 0) {
            best_score = ss().excluded ? alpha : in_check ? mated_in(ply()) : 0;
        }

        const Move prev_move = ss().prev ? ss().prev->position->move() : Move::none();
        {
            int bonus = static_cast<int>(std::pow(static_cast<double>(depth), 1024.0 / m_parameters.bonus_pow_div)) * m_parameters.bonus_depth_mul;
            int malus = depth * m_parameters.malus_depth_mul;

            bonus += (local_best_move == tt_move) * m_parameters.bonus_tt;

            if (ss().position->is_quiet(local_best_move)) {
                for (const auto [m, raise, cutoff, s] : explored_quiets) {
                    int applied{0};
                    if (m == local_best_move) {
                        bonus += raise * m_parameters.bonus_rise;
                        bonus += cutoff * m_parameters.bonus_cutoff;
                        bonus = bonus * 1024 / m_parameters.bonus_mul_div;
                        applied = bonus;
                    } else {
                        malus -= raise * 1024 / m_parameters.malus_mul_div;
                        applied = -malus;
                    }
                    ss().history->at(ss().position(), m) << applied;
                    if (ss().continuation_history) {
                        ss().continuation_history->at(ss().position(), m) << applied;
                    }
                    if (ss().prev && ss().prev->continuation_history)
                        ss().prev->continuation_history->at(ss().position(), m) << applied;
                }
            } else {
                ss().capture_history->at(ss().position(), local_best_move) << bonus ;
            }

            auto explored_captures =
                explored_tacticals | std::views::filter([this](auto e) { return ss().position->is_capture(e.move); });
            for (const auto [m, raise, cutoff, s] : explored_captures) {
                    ss().capture_history->at(ss().position(), m) << -depth;
            }
        }

        TT::Bound bound = (best_score >= beta)                  ? TT::LOWER
                          : (best_score <= alpha_org || !is_pv) ? TT::UPPER
                                                                : TT::EXACT;
        if (!ss().excluded)
            m_tt->store(ss().position->hash(),
                        static_cast<uint16_t>(depth),
                        static_cast<int16_t>(TT::store_score(best_score, ply())),
                        bound,
                        local_best_move,
                        static_cast<int16_t>(ss().static_eval));

        if (alpha != alpha_org && !ss().excluded) {
            ss().best_move = local_best_move;
        }

        return best_score;
    }

    inline int
    SearchThread::QSearch(int alpha, int beta) {
        // assert(beta > -INF && beta < INF);

        if (m_thread_id == 0 && m_statistics.nodes % 4096 == 0) m_tm->update_time();

        const bool              is_pv    = beta - alpha > 1;
        const bool              in_check = ss().position->in_check(ss().position->side_to_move());
        const Positions::Handle pos      = ss().position;

        if (ply() >= MAX_PLY) return evaluate();
        if (is_draw()) return 0;

        const int stand_pat = evaluate();
        ss().static_eval    = stand_pat;

        MoveList moves;
        auto       tt_hit  = m_tt->probe(ss().position->hash());
        const Move tt_move = tt_hit ? tt_hit->move : Move::none();
        if (tt_hit) {
            moves = gen_moves(ss().position());
            if (auto it = ranges::find(moves, tt_move); it == moves.end()) {
                tt_hit = std::nullopt;
            }
        }
        if (tt_hit) {
            if (!ss().position->is_legal(tt_move)) {
                tt_hit = std::nullopt;
            }
        }
        if (tt_move) tt_hit = causes_draw(tt_move) ? std::nullopt : tt_hit;
        const int tt_score = tt_hit ? TT::read_score(tt_hit->score, ply()) : 0;
        if (!is_pv && tt_hit) {
            assert(tt_score > -INF && tt_score < INF);
            if (tt_hit->bound == TT::EXACT || (tt_hit->bound == TT::LOWER && tt_score >= alpha) ||
                (tt_hit->bound == TT::UPPER && tt_score <= beta)) {
                return tt_score;
            }
        }

        int  best_eval = -INF;

        if (!in_check) {
            if (stand_pat >= beta) return beta;
            if (stand_pat > alpha) alpha = stand_pat;
            best_eval = stand_pat;
        }

        if (moves.empty()) {
            moves  = gen_moves(ss().position());
        }
        auto filter = [&](const Move move) {
            if (in_check)
                return pos->is_legal(move);
            else
                return pos->is_tactical(move) && pos->is_legal(move);
        };

        MoveSelector selector{moves | std::views::filter(filter),
                              MoveSelector::Stage::QSearch,
                              ss(),
                              m_parameters.scoring_parameters,
                              tt_move};

        Move best_move  = Move::none();
        int  move_count = 0;
        int  score      = -INF;

        for (const auto [m, s] : selector) {
            if (s < selector.params().good_capture && move_count > 1) {
                continue;
            }

            do_move(m);
            m_tt->prefetch(ss().position->hash());

            m_statistics.nodes++;
            move_count++;

            score = -QSearch(-beta, -alpha);

            undo_move();

            if (m_tm->should_stop()) {
                return 0;
            }

            if (score > best_eval) {
                best_eval = score;
                best_move = m;
            }
            if (best_eval > alpha) alpha = best_eval;
            if (alpha >= beta) break;
        }

        if (move_count == 0) {
            if (in_check) {
                return mated_in(ply());
            }
        }

        assert(best_eval > -INF && best_eval < INF);
        if (best_move != Move::none() && best_move != Move::null()) {
            // TT::Bound bound = (best_eval >= beta) ? TT::LOWER : TT::UPPER;
            //  m_tt->store(make_replacement_policy(), ss().position->hash(), 0, TT::store_score(best_eval, ply()),
            //  bound, best_move, stand_pat);
        }

        return best_eval;
    }

    struct SearchThreadHandler {
        std::size_t m_nb_threads;
        std::vector<std::unique_ptr<SearchThread>> threads{};
        std::vector<std::jthread>                  workers{};

        TimeManager m_tm{};
        TT*         m_tt;

        void
        set(const size_t                                numThreads,
            const SearchThread::Parameters&             params,
            const TimeManager::Params&                  tm_params,
            const TimeManager::UCIConstraints&          tm_constraints,
            TT*                                         tt,
            const Positions&                            pos,
            const nnue::NetworkHandle& network) {
            m_nb_threads = numThreads;
            threads.clear();
            threads.reserve(m_nb_threads);
            workers.clear();
            workers.reserve(m_nb_threads);
            const TimeManager::InitInfo tm_init{.side         = pos.last().side_to_move(),
                                                .moves_played = (pos.last().full_move_clock() / 2),
                                                .static_eval  = 0 /* TODO fix init eval */};
            m_tm = TimeManager{tm_params, tm_init, tm_constraints};
            m_tt = tt;
            for (size_t i = 0; i < m_nb_threads; i++) {
                threads.emplace_back(std::make_unique<SearchThread>(params, i, &m_tm, tt, pos, network.get()));
            }
        }

        std::pair<uint64_t, uint64_t>
        start(const std::stop_token& st, std::function<void()> cb) {
            workers.clear();
            m_tt->new_generation();
            m_tm.start();


            for (const auto& thread : threads) {
                auto t = thread.get();
                workers.emplace_back([t]() {
                    t->IterativeDeepening();
                });
            }

            for (auto& w : workers) {
                w.join();
            }

            auto bestmove = get_best_move();
            std::pair<uint64_t, uint64_t> res{get_total_nodes(), m_tm.elapsed_ms() * m_nb_threads};

            threads.clear();
            workers.clear();

            cb();
            if (bestmove != Move::none()) {
                fmt::println(stdout, "bestmove {}", bestmove);
                std::fflush(stdout);
            }

            return res;
        }

        [[nodiscard]] Move
        get_best_move() const {
            std::unordered_map<Move, int> move_votes;

            for (const auto& t : threads) move_votes[t->root_best_move]++;

            const auto it =
                std::ranges::max_element(move_votes, [](const auto& a, const auto& b) { return a.second < b.second; });

            return it != move_votes.end() ? Move{it->first} : Move{};
        }

        [[nodiscard]] uint64_t
        get_total_nodes() const {
            uint64_t total = 0;
            for (const auto& t : threads) total += t->m_statistics.nodes;
            return total;
        }

        [[nodiscard]] uint64_t
        get_total_nps() const {
            return get_total_nodes() / m_tm.elapsed_ms();
        }

        void
        stop_all() {
            m_tm.stop();
        }
    };
} // namespace chepp

#endif // SEARCHER_H
