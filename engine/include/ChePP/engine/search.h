#ifndef SEARCHER_H
#define SEARCHER_H

#include "history.h"
#include "move_ordering.h"
#include "movegen.h"
#include "nnue.h"
#include "tm.h"
#include "tt.h"

#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "search_stack.h"

inline std::function<int(bool, int, int)> default_lmr = [](const bool quiet, const int d, const int m)
{
    return quiet ? static_cast<int>(0.2 + std::log(m) * std::log(d) / 3.55)
                 : static_cast<int>(1.35 + std::log(m) * std::log(d) / 2.75);
};

inline std::function<int(bool, int)> default_lmp = [](const bool improving, const int d)
{ return improving ? static_cast<int>(4 + 4 * d * d / 4.5) : static_cast<int>(2.5 + 2 * d * d / 4.5); };



struct SearchThread
{
    // can be overridden by UCI default params, and themself overridden by user
    struct Parameters
    {
        int  n_pv{1};
        bool use_syzygy{false};

        int aspiration_window_activation_depth{7};
        int aspiration_window_default_value{50};
        int aspiration_window_multiplicative_factor{2};

        std::function<int(bool, int)>      lmp{default_lmp};
        std::function<int(bool, int, int)> lmr{default_lmr};

        Scorer<ScorerType::Search>::Params search_scorer_params{};
        Scorer<ScorerType::Root>::Params root_scorer_params{};
        Scorer<ScorerType::QSearch>::Params qsearch_scorer_params{};

        int tt_replacement_threshold{3};
    };

    struct Cache
    {
        std::array<std::array<int, MAX_MOVES>, 2>                      lmp{};
        std::array<std::array<std::array<int, MAX_MOVES>, MAX_PLY>, 2> lmr{};
    };

    struct Statistics
    {
        uint64_t nodes{0};
        uint64_t tt_hits{0};
        uint64_t tb_hits{0};
        std::chrono::high_resolution_clock::time_point t_start{};
    };

    struct PvLine
    {
        int32_t                   score{0};
        std::array<Move, MAX_PLY> line{};
    };

    explicit SearchThread(const Parameters& parameters, const int id, TimeManager& tm, TT& tt, const Positions& pos)
        : m_thread_id(id), m_parameters(parameters), m_tm(tm), m_tt(tt), m_search_stack(pos), m_pv_lines(parameters.n_pv)
    {
        init_cache();
    }

    int                   m_thread_id;
    Parameters            m_parameters;
    Cache                 m_cache;
    Statistics            m_statistics;
    TimeManager&          m_tm;
    TT&                   m_tt;
    SearchStack           m_search_stack;
    std::vector<PvLine>   m_pv_lines{};


    void init_cache()
    {
        for (int improving = 0; improving < 2; ++improving)
        {
            for (int d = 1; d < MAX_MOVES; ++d)
            {
                m_cache.lmp[improving][d] = m_parameters.lmp(improving, d);
            }
        }

        for (int quiet = 0; quiet < 2; ++quiet)
        {
            for (int d = 1; d < MAX_PLY; ++d)
            {
                for (int m = 1; m < MAX_MOVES; ++m)
                {
                    m_cache.lmr[quiet][d][m] = m_parameters.lmr(quiet, d, m);
                }
            }
        }
    }

    [[nodiscard]] int                      ply() const { return m_search_stack.ply(); }
    [[nodiscard]] SearchStack::Node&       ss() { return m_search_stack[m_search_stack.ply()]; }
    [[nodiscard]] const SearchStack::Node& ss() const { return m_search_stack[m_search_stack.ply()]; }
    void                                   do_move(const Move move, const bool update_nnue = true) { m_search_stack.do_move(move, update_nnue); }
    void                                   undo_move(const bool update_nnue = true) { m_search_stack.undo_move(update_nnue); }

    int32_t evaluate()
    {
        assert(!ss().position->in_check(ss().position->side_to_move()));
        assert(!is_draw());

        auto eval = ss().accumulator->evaluate(ss().position->side_to_move());
        eval      = std::clamp(eval, LOSS_TB + 1, WIN_TB - 1);
        eval -= eval * ss().position->halfmove_clock() / 101;
        return eval;
    }

    [[nodiscard]] bool is_draw() const
    {
        return ss().is_repetition || ss().position->halfmove_clock() >= 100 ||
               ss().position->is_insufficient_material();
    }

    [[nodiscard]] std::string format_pv_line(const int n) const
    {
        assert(n < m_parameters.n_pv);
        std::ostringstream oss;
        for (const auto m : m_pv_lines[n].line)
        {
            oss << m << " ";
        }
        return oss.str();
    }

    auto make_replacement_policy() const
    {
        return [this] (const TT::Entry& old, const TT::Entry& candidate)
        {
            return candidate.generation != old.generation || candidate.hash != old.hash ||
                (candidate.bound == TT::EXACT && old.bound != TT::EXACT) || candidate.depth > old.depth + m_parameters.tt_replacement_threshold;
        };
    }

    [[nodiscard]] bool causes_draw(const Move move)
    {
        bool ret = false;
        do_move(move, false);
        if (is_draw()) ret = true;
        undo_move(false);
        return ret;
    }


    void IterativeDeepening();
    int  AspirationWindow(int depth, int prev_eval);
    int  Negamax(int depth, int alpha, int beta);
    int  QSearch(int alpha, int beta);
};

inline void SearchThread::IterativeDeepening()
{
    int prev_eval = evaluate();
    m_statistics.t_start = std::chrono::high_resolution_clock::now();

    for (int depth = 1; m_tm.update_depth(depth), !m_tm.should_stop(); ++depth)
    {
        const int eval = AspirationWindow(depth, prev_eval);
        assert(eval > -INF && eval < INF);
        if (!m_tm.should_stop()) // aspiration window got cancelled, we discard the result
        {
            prev_eval = eval;

            if (m_thread_id == 0)
            {
                std::string score;
                if (eval >= MATE_IN_MAX_PLY || eval <= MATED_IN_MAX_PLY)
                {
                    score.append("mate ");
                    score.append(std::to_string((MATE - eval) / 2));
                }
                else
                {
                    score.append("cp ");
                    score.append(std::to_string(eval));
                }

                auto t_now = std::chrono::high_resolution_clock::now();
                auto time_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - m_statistics.t_start);
                int nps = m_statistics.nodes / time_since_start.count();
                std::string uci_output = std::format("info score {} depth {} nodes {} nps {} tb_hits {} pv {}",
                    score, depth, m_statistics.nodes, nps, m_statistics.tb_hits, format_pv_line(0));
                std::cout << uci_output << std::flush;

                TimeManager::UpdateInfo update_info;
                update_info.eval = eval;
                m_tm.adjust_time(update_info);
            }
        }
    }
}

inline int SearchThread::AspirationWindow(const int depth, const int prev_eval)
{
    if (depth < m_parameters.aspiration_window_activation_depth)
    {
        return Negamax(depth, -INF_SCORE, INF_SCORE);
    }

    int window = m_parameters.aspiration_window_default_value;
    int alpha  = prev_eval - window;
    int beta   = prev_eval + window;

    auto eval = Negamax(depth, alpha, beta);

    while (eval <= alpha || eval >= beta)
    {
        if (m_tm.should_stop())
            break;

        window *= m_parameters.aspiration_window_multiplicative_factor;
        alpha = std::clamp(eval - window, -INF_SCORE, INF_SCORE);
        beta  = std::clamp(eval + window, -INF_SCORE, INF_SCORE);

        eval = Negamax(depth, alpha, beta);
    }
    return eval;
}

inline int SearchThread::Negamax(int depth, int alpha, int beta)
{

    if (m_thread_id == 0 && m_statistics.nodes % 4096 == 0)
    {
        m_tm.update_time();
    }

    const Position& pos = *ss().position;

    const int  alpha_org = alpha;
    const bool is_root   = ply() == 0;
    const bool in_check  = pos.checkers(pos.side_to_move()).value();

    assert(depth >= 0);

    // quiescence search supposed to prevent horizon effect
    if (depth <= 0)
        return QSearch(alpha, beta);

    // increase depth if we are in check
    if (in_check)
    {
        depth++;
    }


    m_statistics.nodes++;


    if (!is_root)
    {
        if (is_draw())
        {
            return 0;
        }
        if (ply() >= MAX_PLY)
        {
            return evaluate();
        }

        // this speeds up mate cases
        // our worse move is to be mated on the spot
        alpha = std::max(alpha, mated_in(ply()));
        // their best move is to mate next turn
        beta = std::min(beta, mate_in(ply() + 1));

        if (alpha >= beta)
        {
            return alpha;
        }
    }


    const bool is_pv = beta - alpha > 1;

    // Probe the TT to see if we have a candidate score
    /* IMPORTANT TO REDO AFTER TT CHANGES*/
    auto tt_hit = ss().excluded ? std::nullopt : g_tt.probe(pos.hash());
    if (tt_hit)
    {
        do_move(tt_hit->move);
        if (is_draw())
            tt_hit = std::nullopt;
        undo_move();
    }
    if (!is_pv && tt_hit)
    {
        const TT::Entry& e = *tt_hit;
        if (e.depth >= depth)
        {
            const int score = TT::read_score(e.score, ply());
        if (e.bound == TT::Bound::EXACT || (e.bound == TT::Bound::LOWER && score >= alpha) || (e.bound == TT::Bound::UPPER && score <= beta))
            {
                m_statistics.tt_hits++;
                return score;
            }
        }
    }

    // Check for tablebases
    if (!is_root && pos.occupancy().popcount() <= 7)
    {
        auto wdl = pos.wdl_probe();
        if (wdl != TB_RESULT_FAILED)
        {
            m_statistics.tb_hits++;

            int score;
            switch (wdl)
            {
                case TB_LOSS:
                    score = LOSS_TB + ply();
                    break;
                case TB_DRAW:
                case TB_BLESSED_LOSS:
                case TB_CURSED_WIN:
                    score = 0;
                    break;
                case TB_WIN:
                    score = WIN_TB - ply();
                    break;
                default:
                    score = 0;
            }

            return score;
        }
    }

    if (false && is_root && pos.occupancy().popcount() <= 7)
    {
        auto res = pos.dtz_probe();
        if (res != TB_RESULT_FAILED)
        {
            m_statistics.tb_hits++;

            switch (res)
            {
                case TB_LOSS:
                    break;
                case TB_DRAW:
                    break;
                case TB_WIN:
                {
                    Square from{TB_GET_FROM(res)};
                    Square to{TB_GET_TO(res)};
                    Square ep{TB_GET_EP(res)};
                    Piece  promote{TB_GET_PROMOTES(res)};

                    break;
                }

                default:
                    break;
            }
        }
    }

    int static_eval = in_check ? 0 : tt_hit ? tt_hit->score : evaluate(); // Important static_eval is 0 if in check
    assert(static_eval > -INF);

    // careful need to manage eval properly
    ss().eval = static_eval;


    // the improving heuristic, basically checks if the sequence of moves improves the position
    // used to be more cautious of fail low, less cautious of fail highs in futility prunings
    bool is_improving = in_check        ? false
                        : ss().ply >= 4 ? ss().prev->prev->prev->prev->static_eval > static_eval
                        : ss().ply >= 2 ? ss().prev->prev->static_eval > static_eval
                                        : true;


    // testing reverse futility pruning, basically if the evaluation is already crazy high, just fail high the node
    // need to be careful though because can give the illusion of strong moves to the search tree, which is the reason
    // for the adjustment of the search score
    if (!is_root && !is_pv && !in_check && depth < 9 &&
        static_eval >= beta + ((depth - is_improving) * 77 - ss().prev->eval / 400))
    {
        return static_eval;
    }

    // null move pruning
    // idea : if we expect we will beat beta, we offer a free move and search at reduced depth
    // if eval comes from tt, is upper bounded and not higher that beta, we cant assume anything on score
    // evaluating is not worth it so we just skip
    // only do it if there are enough pieces to not avoid zugzwang blindness

    // to do inm case of anexcluded move that comes ffrom resear5ch where we retry without the tt
    if (!is_root && !is_pv && ss().position->move() != Move::null() && !in_check && depth >= 3 && static_eval >= beta &&
        (!tt_hit || tt_hit->bound != TT::Bound::UPPER || tt_hit->score > beta) && std::abs(static_eval) < MATE_IN_MAX_PLY &&
        pos.occupancy(KNIGHT, BISHOP, ROOK, QUEEN).popcount() >= 3) // add loss condition ?
    {
        const int reduction  = 3 + depth / 3 + std::clamp((static_eval - beta) / 100, 0, 4);
        int       null_depth = std::max((depth - 1) / 2, (depth - reduction - 1) / 2);
        do_move(Move::null());

        assert(null_depth >0); // do not drop in qsearch
        auto score = -Negamax(null_depth, -beta, -(beta - 1));

        undo_move();

        if (score >= beta)
        {
            if (std::abs(score) >= MATE_IN_MAX_PLY)
            {
                score = beta;
            }
            return score;
        }
    }

    // generate all legal moves
    MoveList moves = gen_legal(pos);

    if (moves.empty())
    {
        return in_check ? mated_in(ply()) : 0;
    }

    ScoredMoveList scored_moves;
    if (is_root && depth > 7)
    {
        Scorer<ScorerType::Root> search_scorer(m_parameters.root_scorer_params, ss(), tt_hit ? tt_hit->move : Move::none());
        scored_moves = make_scored_list(moves, search_scorer());
    } else {
        Scorer<ScorerType::Search> search_scorer(m_parameters.search_scorer_params, ss(), tt_hit ? tt_hit->move : Move::none());
        scored_moves = make_scored_list(moves, search_scorer());
    }

    // probcut, need to look at conditions and parameters more closely
    if (!is_root && !ss().excluded && !is_pv && !in_check && depth >= 3 && static_eval >= beta + 150)
    {
        int prob_beta = beta + 150;

        auto tactical = moves | std::views::filter(make_tactical_predicate(pos));
        Scorer<ScorerType::QSearch> search_scorer(m_parameters.qsearch_scorer_params, ss(), tt_hit ? tt_hit->move : Move::none());

        for (auto [move, s]: make_scored_list(tactical, search_scorer()))
        {
            if (move == tt_hit->move || s < -1'000'000ULL) //check the scaling
            {
                continue;
            }
            do_move(move);

            auto score = -QSearch(-prob_beta, -(prob_beta - 1));

            if (score >= prob_beta)
            {
                const int reduction  = 3;
                int       prob_depth = std::max(1, depth - 1 - reduction);
                prob_beta            = -Negamax(prob_depth, -beta, -beta + 1);
            }

            undo_move();

            if (score >= prob_beta)
            {
                return score;
            }
        }
    }

    int  best_eval   = -INF_SCORE;
    Move local_best  = Move::none();
    bool first_move  = true;
    int  move_idx    = 0;
    bool skip_quiets = false;
    int  score       = -INF_SCORE;

    MoveList quiets{};
    MoveList captures{};

    // Move loop
    // CHECK THE ORDERING
    for (auto [m, s] : scored_moves)
    {

        if (m == ss().excluded)
        {
            assert(moves.size() > 1);
            continue;
        }

        assert((move_idx==0 && first_move) || (move_idx > 0 && !first_move));

        bool is_quiet = !pos.is_occupied(m.to_sq()) && m.type_of() != EN_PASSANT && m.type_of() != PROMOTION;
        if (is_quiet)
            quiets.push_back(m);
        bool is_captured = pos.is_occupied(m.to_sq()) || m.type_of() == EN_PASSANT;
        if (is_captured)
            captures.push_back(m);

        // Some pruning
        if (!is_root && best_eval > MATED_IN_MAX_PLY && local_best != Move::none())
        {
            // Pruning for quiets

            int lmrDepth = m_cache.lmr.at(is_quiet).at(depth).at(move_idx);

            if (is_quiet)
            {
                // Pruning barbare need to be sure
                if (skip_quiets)
                {
                    move_idx++;
                    first_move = false;
                    continue;
                }

                // Late Move Pruning. Relies on effective ordering of the moves.
                // Reached if a certain number of quiet moves has been reached.
                // Then ignore the following ones.
                if (!is_pv && !in_check && depth <= 7)
                {
                    if (quiets.size() > m_cache.lmp.at(is_improving).at(move_idx))
                    {
                        skip_quiets = true; // skip this node continue the search now skipping quiets
                        move_idx++;
                        first_move = false;
                        continue;
                    }
                }

                // Continuation pruning.
                //  Weird but slos down the search at least in some position
                if (lmrDepth < 3 && ss().history.get_bonus(m) < -4'000 * depth) // CHECK SCALING
                {
                    move_idx++;
                    first_move = false;
                    continue;
                }

                // Futility Pruning, probably needs nore conditions
                if (!is_pv && !in_check && lmrDepth <= 6)
                {
                    const int margin = futility_margin_for_depth(depth);
                    if (static_eval + margin + 100 * is_improving <= alpha)
                    {
                        skip_quiets = true; // skip this node continue the search now skipping quiets
                        move_idx++;
                        first_move = false;
                        continue;
                    }
                }

                // SEE pruning for quiets. Approximate of the rice implementation, need to change see computation
                if (depth <= 8 && is_captured && pos.see(m) + 70 * depth < 0)
                {
                    move_idx++;
                    first_move = false;
                    continue;
                }
            }
            else
            {
                // SEE pruning but for noisy
                if (depth <= 6 && is_captured && pos.see(m) + 15 * depth * depth < 0)
                {
                    move_idx++;
                    first_move = false;
                    continue;
                }
            }
        }

        int search_depth = depth;

        uint64_t begin = m_statistics.nodes;

        bool allow_singular_extension = false;
        bool double_extend            = false;
        bool negative_extension       = false;
        Move tt_move                  = tt_hit ? tt_hit->move : Move::none();

        // Extend the search if the move comes from TT.
        if (!is_root && !is_pv && depth >= 6 && tt_move != Move::none() && (tt_hit->bound == TT::Bound::LOWER || tt_hit->bound == TT::Bound::EXACT) &&
            tt_hit->depth >= depth - 3 && std::abs(TT::read_score(tt_hit->score, ply())) < MATE_IN_MAX_PLY &&
            moves.size() > 1)
        {
            int tt_score       = TT::read_score(tt_hit->score, ply());
            int singular_beta  = tt_score - depth;
            int singular_depth = (depth - 1) / 2;
            assert(singular_depth > 0);
            ss().excluded      = tt_move;
            int singular_score = Negamax(singular_depth, singular_beta - 1, singular_beta);
            ss().excluded      = Move::none();
            assert(singular_beta > MATED);
            if (singular_score < singular_beta)
            {
                allow_singular_extension = true;

                if (singular_score < singular_beta - 20 && ss().double_extensions <= 5)
                {
                    double_extend          = true;
                    ss().double_extensions = ss().prev() ? ss().prev()->double_extensions + 1 : 1;
                }
            }
            else if (tt_score >= beta)
            {
                return tt_score;
            }
            else if (tt_score <= singular_score || !is_pv) /* TODO should we negative extend ALL non PV nodes? */
            {
                negative_extension = true;
            }
        }

        if (m == tt_move)
        {
            if (allow_singular_extension)
            {
                search_depth += 1;
                if (double_extend)
                    search_depth += 1;
            }
            else if (negative_extension)
            {
                search_depth = std::max(1, search_depth - 1);
            }
        }

        do_move(m);

        bool fullsearch = !is_pv || move_idx > 0;

        // LMR. Moves that are late enough are searched at reduced depth depending on factors.
        // If they beat alpha, they are researched full depth but reduced window.
        if (depth >= 3 && !in_check && move_idx > 2 * (1 + is_pv) &&
            (true || !allow_singular_extension) /* TODO should we reduce singular moves ?) */)
        {
            int reduction = std::min(m_cache.lmr.at(is_quiet).at(depth).at(move_idx), depth - 1);

            reduction += !is_improving; // Increase the reduction for non improvment
            reduction += !is_pv;        // Increase reduction if non PV
            // Should add a reduction for quiet moves that lose material , e.g if the quiet move leaves us open to a
            // take reduction += is_quiet;

            // reduction -= m_history.get_hist_score(ss(), m) / 4'000; // Reduce or increase depending on history score
            // /* TODO fix scaling  rn it just sets it to 1 or max*/
            reduction -= 2 * (m == ss().killer1 || m == ss().killer2); // Reduce if the move is killer

            // adjustment to avoid dropping into a Qsearch.
            reduction = std::min(depth - 2, std::max(reduction, 1));
            search_depth -= reduction;

            assert(search_depth > 0);
            // do the search at reduced depth (picking up from where the extensions left us)
            score = -Negamax(search_depth - 1, -alpha - 1, -alpha);
            assert(score != -INF);

            // go full depth if score beat alpha
            fullsearch = score > alpha && reduction != 1;

            // go deeper on the full search in case the beats by a margin.
            // Recall that search_depth is the new depth based on the extensions.
            bool deeper = score > best_eval + 70 + 12 * (search_depth - reduction);

            search_depth += deeper;
        }

        // Full depth null window
        if (fullsearch)
        {
            score = -Negamax(search_depth - 1, -alpha - 1, -alpha);
            assert(score != -INF);
        }

        // PVS
        if (is_pv && (first_move || (score > alpha && score < beta)))
        {
            score = -Negamax(search_depth - 1, -beta, -alpha);
            assert(score != -INF);
        }

        undo_move();

        uint64_t end = m_statistics.nodes;
        if (is_root)
        {
            assert(ss().refutation_nodes);
            ss().refutation_nodes->at(m) += end - begin;
        }

        // if we out of time we just return 0 and it will be discarded down the line
        if (m_tm.should_stop())
        {
            return 0;
        }

        if (score > best_eval)
        {
            best_eval  = score;
            local_best = m;
        }
        if (score > alpha)
            alpha = score;

        if (alpha >= beta)
        {
            if (is_quiet)
            {
                if (ss().killer1 != m)
                {
                    ss().killer2 = ss().killer1;
                    ss().killer1 = m;
                }
                m_history.update_cont_hist(ss(), quiets, m, depth);
                m_history.update_hist(ss(), quiets, m, depth);
                m_history.update_pawn_hist(ss(), quiets, m, depth);
            }
            if (is_captured)
            {
                m_history.update_capture_hist(ss(), captures, m, depth);
            }
            assert(local_best != Move::none());
            break;
        }

        first_move = false;
        move_idx++;

        // here the search result is actually valid, and since we searched pv node first,
        // we can accept the result as valid
        if (m_tm.should_stop() && is_root && local_best != Move::none())
        {
            break;
        }
    }

    if (m_thread_id == 0 && is_root)
    {
        TimeManager::UpdateInfo info{};
        info.eval           = absolute_eval(best_eval, pos.side_to_move());
        info.nodes_searched = m_statistics.nodes;
        m_tm.send_update_info(info);
    }

    if (local_best == Move::none())
    {
        std::cout << std::format("local best {}", best_eval) << std::endl;
        throw new std::runtime_error("");
    }

    assert(local_best != Move::none() && local_best != Move::null());
    bool best_valid = !m_tm.should_stop() && local_best != Move::none() && ss().excluded == Move::none();
    if (is_root && best_valid)
        best_move = local_best;

    // std::cout << best_valid << " " << local_best << " " << best_eval << " " << evaluate() << std::endl;

    TT::Bound bound = (best_eval <= alpha_org) ? bound = TT::UPPER : (best_eval >= beta) ? TT::LOWER : TT::EXACT;
    if (best_valid)
        g_tt.store( make_replacement_policy() ,pos.hash(), depth, store_tt_score(best_eval, ply()), bound, local_best);

    assert(best_eval > -INF && best_eval < INF);
    return best_eval;
}

inline int SearchThread::QSearch(int alpha, int beta)
{
    assert(beta > -INF && beta < INF);

    if (m_thread_id == 0 && m_statistics.nodes % 4096 == 0) m_tm.update_time();
    m_statistics.nodes++;

    const Position& pos = *ss().position;
    const bool is_pv = beta - alpha > 1;
    const bool in_check = pos.in_check(pos.side_to_move());


    if (ply() >= MAX_PLY)
        return evaluate();

    if (is_draw())
        return 0;

    auto filter = [&] (const Move move )
    {
        return in_check ?
            make_legal_predicate(pos)(move) : // resolve checks
            make_legal_predicate(pos)(move) && make_tactical_predicate(pos)(move); //only take captures/promotions
    };

    const auto moves = gen_legal(pos);
    if (moves.empty()) return in_check ? mated_in(ply()) : 0;
    const auto tactical = moves | std::views::filter(filter);



    auto tt_hit = m_tt.probe(pos.hash());
    tt_hit = causes_draw(tt_hit->move) ? std::nullopt : tt_hit;
    const Move tt_move = tt_hit ? tt_hit->move : Move::none();
    if (!is_pv && tt_hit)
    {
        const auto& e     = *tt_hit;
        const int         score = TT::read_score(e.score, ply());
        assert(score > -INF && score < INF);
        if (e.bound == TT::EXACT || (e.bound == TT::LOWER && score >= alpha) || (e.bound == TT::UPPER && score <= beta)) {
            return score;
        }
    }

    if (pos.occupancy().popcount() <= 7)
    {
        auto wdl = pos.wdl_probe();
        if (wdl != TB_RESULT_FAILED)
        {
            m_statistics.tb_hits++;

            int score;
            switch (wdl)
            {
                case TB_LOSS:
                    score = LOSS_TB + ply();
                    break;
                case TB_DRAW:
                case TB_BLESSED_LOSS:
                case TB_CURSED_WIN:
                    score = 0;
                    break;
                case TB_WIN:
                    score = WIN_TB - ply();
                    break;
                default:
                    score = 0;
            }

            return score;
        }
    }

    const int stand_pat = evaluate();
    ss().static_eval           = stand_pat;

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    Scorer<ScorerType::QSearch> scorer(m_parameters.qsearch_scorer_params, ss(), tt_move);
    ScoredMoveList scored_moves = make_scored_list(moves, scorer());


    int  best_eval = stand_pat;
    Move best_move = Move::none();
    for (const auto [m, s] : make_scored_list(tactical, scorer()))
    {
        // std::cout << m << std::endl;
        if (!is_pv && pos.is_occupied(m.to_sq()) &&
            ((s < -5'000'000) || pos.piece_at(m.to_sq()).piece_value() + 2 * s + best_eval <
                                     alpha)) // see pruning on captures, we don't want to look at hopeless captures
        {
            continue;
        }

        do_move(m);

        const int score = -QSearch(-beta, -alpha);

        undo_move();

        if (m_tm.should_stop())
        {
            break;
        }

        if (score > best_eval)
        {
            best_eval = score;
            best_move = m;
        }
        if (best_eval > alpha)
            alpha = best_eval;
        if (alpha >= beta)
            break;
    }
    tt_bound_t bound = (best_eval >= beta) ? LOWER : UPPER;
    if (best_move != Move::none())
        g_tt.store(pos.hash(), 0, store_tt_score(best_eval, ply()), bound, best_move);
    assert(best_eval > -INF && best_eval < INF);
    return best_eval;
}

struct SearchThreadHandler
{
    std::vector<std::unique_ptr<SearchThread>> threads{};
    std::vector<std::jthread>                  workers{};
    TimeManager                                m_tm{};

    void set(const size_t numThreads, const SearchThread::Parameters& params, const TimeManager& tm, const Positions& pos)
    {
        threads.clear();
        threads.reserve(numThreads);
        workers.clear();
        workers.reserve(threads.size());
        m_tm = tm;
        for (size_t i = 0; i < numThreads; i++)
        {
            threads.push_back(std::make_unique<SearchThread>(params, i, m_tm, pos));
        }
    }

    void start()
    {
        g_tt.new_generation();

        m_tm.start();

        for (const auto& thread : threads)
        {
            workers.emplace_back([t = thread.get()]() { t->IterativeDeepening(); });
        }

        for (auto& w : workers)
            if (w.joinable())
                w.join();

        if (const auto move = get_best_move(); move != Move::none())
        {
            std::cout << "bestmove " << move << std::endl;
        }

        threads.clear();
        workers.clear();
    }

    [[nodiscard]] Move get_best_move() const
    {
        std::unordered_map<uint16_t, int> move_votes;

        for (const auto& t : threads)
        {
            move_votes[t->m_pv_lines[0][0].raw()]++;
        }

        const auto it =
            std::ranges::max_element(move_votes, [](const auto& a, const auto& b) { return a.second < b.second; });

        return it != move_votes.end() ? Move{it->first} : Move{};
    }

    void stop_all() { m_tm.stop(); }
};

#endif // SEARCHER_H
