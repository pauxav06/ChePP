#ifndef SEARCHER_H
#define SEARCHER_H

#include "move_ordering.h"
#include "nnue.h"
#include "tm.h"
#include "tt.h"
#include "history.h"

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


struct SearchThread
{
    struct SearchResult
    {
        int score{};
        int depth{};
        Move best_move{Move::null()};
        bool full_search{};
    };

    struct SearchInfos
    {
        uint64_t nodes;
        uint64_t tt_hits;
        uint64_t tb_hits;
    };

    explicit SearchThread(const int id, TimeManager& tm, const Position& pos, std::span<Move> moves)
        : m_thread_id(id), m_tm(tm), m_positions(pos, moves), m_accumulators(m_positions.last()), m_ss(MAX_PLY + 1), m_root_refutation_time()
    {
        ss().pos = &m_positions.last();
    }

    int          m_thread_id;
    TimeManager& m_tm;

    Positions                          m_positions;
    Accumulators                       m_accumulators;
    SearchStack                        m_ss;

    SearchInfos    m_infos{};
    HistoryManager m_history{};

    std::unordered_map<uint16_t, std::size_t> m_root_refutation_time;

    Move bestMove;


    [[nodiscard]] int ply() const { return static_cast<int>(m_positions.ply()); }
    [[nodiscard]] SearchStack::Node& ss() { return m_ss[ply()]; }

    template <bool UpdateNNUE = true>
    void do_move(const Move move)
    {
        m_positions.do_move(move);
        if constexpr (UpdateNNUE) m_accumulators.do_move(m_positions[ply() - 1], m_positions.last());
        ss().pos = &m_positions.last();
    }

    template <bool UpdateNNUE = true>
    void undo_move()
    {
        ss().pos = nullptr;
        m_positions.undo_move();
        if constexpr (UpdateNNUE) m_accumulators.undo_move();
    }

    int32_t evaluate()
    {
        auto eval = m_accumulators.last().evaluate(m_positions.last().side_to_move());
        eval      = std::clamp(eval, MATED_IN_MAX_PLY + 1, MATE_IN_MAX_PLY - 1);
        eval -= eval * m_positions.last().halfmove_clock() / 101;
        return eval;
    }

    [[nodiscard]] bool is_draw() const { return m_positions.is_repetition() || m_positions.last().is_insufficient_material(); }

    std::span<const Position> positions() { return m_positions.positions(); }

    SearchResult IterativeDeepening();
    int  AspirationWindow(int depth, int prev_eval);
    int  Negamax(int depth, int alpha, int beta);
    int  QSearch(int alpha, int beta);
};

inline std::vector<Move> get_pv_line(const Position& pos, int max_depth = MAX_PLY)
{
    std::vector<Move> pv;
    Position          temp_pos = pos;

    for (int ply = 0; ply < max_depth; ++ply)
    {
        const auto tt_hit = g_tt.probe(temp_pos.hash());
        if (!tt_hit || tt_hit->m_move == Move::none())
            break;

        pv.push_back(tt_hit->m_move);
        temp_pos.do_move(tt_hit->m_move);

        if (gen_legal(temp_pos).empty())
            break;
    }

    return pv;
}

inline void print_pv_line(const Position& pos, const int depth, const int eval)
{
    auto pv = get_pv_line(pos, depth);
    std::cout << "PV (Eval " << eval << "): ";
    for (auto m : pv)
    {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}

inline const std::array<std::array<int, 256>, MAX_PLY>& lmr_table(bool quiet)
{
    static std::array<std::array<int, 256>, MAX_PLY> g_quiet_table = []()
    {
        std::array<std::array<int, 256>, MAX_PLY> lmr{};
        for (int d = 1; d < MAX_PLY; ++d)
        {
            for (int m = 1; m < 256; ++m)
            {
                lmr[d][m] = static_cast<int>(0.2 + std::log(m) * std::log(d) / 3.55);
            }
        }
        return lmr;
    }();
    static std::array<std::array<int, 256>, MAX_PLY> g_noisy_table = []()
    {
        std::array<std::array<int, 256>, MAX_PLY> lmr{};
        for (int d = 1; d < MAX_PLY; ++d)
        {
            for (int m = 1; m < 256; ++m)
            {
                lmr[d][m] = static_cast<int>(1.35 + std::log(m) * std::log(d) / 2.75);
            }
        }
        return lmr;
    }();
    return quiet ? g_quiet_table : g_noisy_table;
}

inline const std::array<int, MAX_PLY>& lmp_table(bool improving)
{
    static std::array<int, MAX_PLY> g_quiet_table = []()
    {
        std::array<int, MAX_PLY> lmp{};
        for (int d = 1; d < MAX_PLY; ++d)
        {
            lmp[d] = static_cast<int>(4 + 4 * d * d / 4.5);
        }
        return lmp;
    }();
    static std::array<int, MAX_PLY> g_noisy_table = []()
    {
        std::array<int, MAX_PLY> lmp{};
        for (int d = 1; d < MAX_PLY; ++d)
        {
            lmp[d]= static_cast<int>(2.5 + 2 * d * d / 4.5);
        }
        return lmp;
    }();
    return improving ? g_quiet_table : g_noisy_table;
}

inline constexpr int FUTILITY_DEPTH_MAX   = 3;
inline constexpr int FUTILITY_BASE_MARGIN = 100;
inline constexpr int FUTILITY_DEPTH_SCALE = 120;

inline int futility_margin_for_depth(int depth)
{
    depth = std::max(1, std::min(depth, MAX_PLY));
    return FUTILITY_BASE_MARGIN + FUTILITY_DEPTH_SCALE * depth;
}

inline SearchThread::SearchResult SearchThread::IterativeDeepening()
{
    int prev_eval = evaluate();

    SearchResult ret;

    int depth = 1;
    for (; m_tm.update_depth(depth), !m_tm.should_stop(); ++depth)
    {
        const auto eval = AspirationWindow(depth, prev_eval);
        if (!m_tm.should_stop())
        {
            prev_eval = eval;

            if (m_thread_id == 0)
            {
                std::string score;
                if (eval >= MATE_IN_MAX_PLY)
                {
                    score.append("mate in ");
                    score.append(std::to_string(MATE - eval));
                }
                else
                    score = std::to_string(eval);

                std::cout << "Depth " << depth << " Eval " << score << " Nodes " << m_infos.nodes << " best "
                          << bestMove << std::endl;
                print_pv_line(m_positions.last(), depth, prev_eval);
            }
        }
    }

    ret.depth = depth - 1;
    ret.best_move = bestMove;
    ret.full_search = false;
    ret.score = prev_eval;
    return ret;
}

struct AspirationStats {
    double variance = 10000.0;
    const double lambda = 0.95;
    int z = 2;

    [[nodiscard]] int window() const {
        double sigma = std::sqrt(variance);
        int w = int(z * sigma);
        if (w < 8) w = 8;
        if (w > 300) w = 300;
        return w;
    }

    void update(int delta_eval) {
        double d2 = double(delta_eval) * double(delta_eval);
        variance = lambda * variance + (1.0 - lambda) * d2;
    }
};

inline int SearchThread::AspirationWindow(const int depth, const int prev_eval)
{
    static AspirationStats stats;
    int alpha, beta;

    if (depth <= 7) {
        alpha = -INF_SCORE;
        beta  = +INF_SCORE;
        auto eval = Negamax(depth, alpha, beta);

        if (depth > 1) {
            stats.update(eval - prev_eval);
        }

        return eval;
    }

    int window = stats.window();
    std::cout <<  "window " << window << std::endl;
    alpha = prev_eval - window;
    beta  = prev_eval + window;

    auto eval = Negamax(depth, alpha, beta);

    while (eval <= alpha || eval >= beta) {
        if (m_tm.should_stop())
            break;

        window *= 2;
        alpha = std::clamp(eval - window, -INF_SCORE, INF_SCORE);
        beta  = std::clamp(eval + window, -INF_SCORE, INF_SCORE);

        eval = Negamax(depth, alpha, beta);
    }

    stats.update(eval - prev_eval);

    return eval;
}


inline auto store_tt_score(const int score, const int ply)
{
    if (score >= MATE_IN_MAX_PLY)
        return score - ply;
    if (score <= MATED_IN_MAX_PLY)
        return score + ply;
    return score;
};

inline auto read_tt_score(const int score, const int ply)
{
    if (score >= MATE_IN_MAX_PLY)
        return score - ply;
    if (score <= MATED_IN_MAX_PLY)
        return score + ply;
    return score;
};

enum SearchNode
{
    Pv,
    Cut,
    All
};

inline int SearchThread::Negamax(int depth, int alpha, int beta)
{

    if (m_thread_id == 0 && m_infos.nodes % 4096 == 0)
    {
        TimeManager::UpdateInfo info{};
        m_tm.update_time();
    }
    const Position&        pos = m_positions.last();

    const int  alpha_org = alpha;
    const bool is_root   = ply() == 0;
    const bool in_check  = pos.checkers(pos.side_to_move()).value();

    // increase depth if we are in check
    depth += in_check;

    // quiescence search supposed to prevent horizon effect
    if (depth <= 0)
        return QSearch(alpha, beta);

    m_infos.nodes++;

    if (!is_root)
    {
        if (is_draw())
        {
            //std::cout << "draw bz rep or insufficient material" << std::endl;
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

    // try to use the TT
    auto tt_hit = ss().excluded ? std::nullopt : g_tt.probe(pos.hash());
    if (tt_hit)
    {
        do_move<false>(tt_hit->m_move);
        if (is_draw())
            tt_hit = std::nullopt;
        undo_move<false>();

    }
    if (!is_pv && tt_hit)
    {
        const tt_entry_t& e = *tt_hit;
        if (e.m_depth >= depth)
        {
            const int score = read_tt_score(e.m_score, ply());
            if (e.m_bound == EXACT || (e.m_bound == LOWER && score >= alpha) || (e.m_bound == UPPER && score <= beta))
            {
                m_infos.tt_hits++;
                return score;
            }
        }
    }


    int static_eval = in_check? 0 : tt_hit ? tt_hit->m_score : evaluate();
    assert(static_eval > -INF);

    ss().eval = static_eval;

    // the improving heuristic, basically checks if the sequence of moves improves the position
    // used to be more cautious of fail low, less cautious of fail highs in futility prunings
    bool is_improving;

    if (in_check)
    {
        is_improving = false;
    }
    else if (ply() >= 4)
    {
        is_improving = ss().prev()->prev()->prev()->prev()->eval > static_eval;
    }
    else if (ply() >=2 )
    {
        is_improving = ss().prev()->prev()->eval > static_eval;
    }
    else
    {
        is_improving = true;
    }

    // testing reverse futility pruning, basically if the evaluation is already crazy high, just fail high the node
    // need to be careful though because can give the illusion of strong moves to the search tree, which is the reason for
    // the adjustment of the search score
    if (!is_root && !is_pv && !in_check && depth < 9 && static_eval >= beta + ((depth - is_improving) * 77 - ss().prev()->eval/400))
    {
        return static_eval;
    }

    // null move pruning
    // idea : if we expect we will beat beta, we offer a free move and search at reduced depth
    // if eval comes from tt, is upper bounded and not higher that beta, we cant assume anything on score
    // evaluating is not worth it so we just skip
    // only do it if there are enough pieces to not avoid zugzwang blindness
    if (!is_root && !is_pv && ss().pos->move() != Move::null() && !in_check && depth >= 3 && static_eval >= beta &&
        (!tt_hit || tt_hit->m_bound != UPPER || tt_hit->m_score > beta) && std::abs(static_eval) < MATE_IN_MAX_PLY &&
        pos.occupancy(KNIGHT, BISHOP, ROOK, QUEEN).popcount() >= 3) // add loss condition ?
    {
        const int reduction = 3 + depth / 3 + std::clamp((static_eval - beta) / 100, 0, 4);
        int null_depth = std::max((depth - 1) / 2, (depth - reduction - 1) / 2);
        do_move<false>(Move::null());

        auto score = -Negamax(null_depth, -beta, -(beta - 1));

        undo_move<false>();

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

    // score the moves to sort them
    if (is_root && depth > 7)
    {
        for (auto& [m, s] : moves)
        {
            s += m_root_refutation_time[m.raw()];
            if (tt_hit && m == tt_hit->m_move)
            {
                s = std::numeric_limits<int>::max();
            }
        }

    } else
    {
        score_moves(ss(), moves, tt_hit ? tt_hit->m_move : Move::none(), m_history, ss());
    }
    moves.sort();

    // probcut, need to look at conditions and parameters more closely
    if (!is_root && !ss().excluded && !is_pv && !in_check && depth >= 3 && static_eval >= beta + 150)
    {
        int       prob_beta = beta + 150;

        MoveList  tactical  = filter_tactical(pos, gen_legal(pos));
        score_moves(ss(), tactical, tt_hit ? tt_hit->m_move : Move::none(), m_history, ss());
        tactical.sort();

        for (auto [m, s] : tactical)
        {
            if (m == tt_hit->m_move || s < -1'000'000)
            {
                continue;
            }
            do_move(m);

            auto score = -QSearch(-prob_beta, -prob_beta + 1);

            if (score >= prob_beta)
            {
                const int reduction  = 3;
                int prob_depth = std::max(1, depth - 1 - reduction);
                prob_beta = -Negamax(prob_depth, -beta, -beta + 1);
            }


            undo_move();

            if (score >= prob_beta)
            {
                return score;
            }
        }
    }


    int      best_eval   = -INF_SCORE;
    Move     local_best  = Move::none();
    bool     first_move  = true;
    int      move_idx    = 0;
    bool     skip_quiets = false;
    int score = -INF_SCORE;


    MoveList quiets{};
    MoveList captures{};

    // Move loop
    for (auto [m, s] : moves)
    {

        if (m == ss().excluded)
        {
            assert(moves.size() > 1);
            continue;
        }

        bool is_quiet = !pos.is_occupied(m.to_sq()) && m.type_of() != EN_PASSANT && m.type_of() != PROMOTION;
        if (is_quiet)
            quiets.push_back(m);
        bool is_captured = pos.is_occupied(m.to_sq()) || m.type_of() == EN_PASSANT;
        if (is_captured)
            captures.push_back(m);



        // Some pruning
        if (!is_root && best_eval > MATED && local_best != Move::none())
        {
            //Pruning for quiets

            int lmrDepth = lmr_table(is_quiet)[depth][move_idx];

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
                    if (quiets.size() > lmp_table(is_improving)[depth])
                    {
                        skip_quiets = true; // skip this node continue the search now skipping quiets
                        move_idx++;
                        first_move = false;
                        continue;
                    }
                }

                //Continuation pruning.
                // Weird but slos down the search at least in some position
                if (false && lmrDepth < 3 && m_history.get_hist_score(ss(), m) < -4'000 * depth)
                {
                    move_idx++;
                    first_move = false;
                    continue;
                }

                //Futility Pruning, probably needs nore conditions
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
                if (depth <= 8 && is_captured && pos.see(m) + 70 * depth <  0)
                {
                    move_idx++;
                    first_move = false;
                    continue;
                }

            } else
            {
                // SEE pruning but for noisy
                if (depth <= 6 && is_captured && pos.see(m) + 15 * depth * depth <  0)
                {
                    move_idx++;
                    first_move = false;
                    continue;
                }
            }
        }



        int search_depth = depth;

        uint64_t begin = m_infos.nodes;


        bool allow_singular_extension = false;
        bool double_extend = false;
        bool negative_extension = false;
        Move tt_move = tt_hit ? tt_hit->m_move : Move::none();


        // Extend the search if the move comes from TT.
        if (!is_root && !is_pv && depth >= 6 && tt_move != Move::none() &&
            tt_hit->m_bound == LOWER && tt_hit->m_depth >= depth - 3 &&
            std::abs(read_tt_score(tt_hit->m_score, ply())) < MATE_IN_MAX_PLY && moves.size() > 1)
        {
            int tt_score = read_tt_score(tt_hit->m_score, ply());
            int singular_beta = tt_score - depth;
            int singular_depth = (depth - 1) / 2;

            ss().excluded = tt_move;
            int singular_score = Negamax(singular_depth, singular_beta - 1, singular_beta);
            ss().excluded = Move::none();

            if (singular_score < singular_beta)
            {
                allow_singular_extension = true;

                if (singular_score < singular_beta - 20 && ss().double_extensions <= 5)
                {
                    double_extend = true;
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
        if (depth >= 3 && !in_check && move_idx > 2 * (1 + is_pv) &&  (true || !allow_singular_extension) /* TODO should we reduce singular moves ?) */)
        {
            int reduction = std::min(lmr_table(is_quiet)[depth][move_idx], depth - 1);

            reduction += !is_improving; // Increase the reduction for non improvment
            reduction += !is_pv; // Increase reduction if non PV
            //Should add a reduction for quiet moves that lose material , e.g if the quiet move leaves us open to a take
            //reduction += is_quiet;

            //reduction -= m_history.get_hist_score(ss(), m) / 4'000; // Reduce or increase depending on history score /* TODO fix scaling  rn it just sets it to 1 or max*/
            reduction -= 2 * (m == ss().killer1 || m == ss().killer2); // Reduce if the move is killer

            //adjustment to avoid dropping into a Qsearch.
            reduction = std::min(depth -1, std::max(reduction, 1));
            search_depth -= reduction;

            // do the search at reduced depth (picking up from where the extensions left us)
            score = -Negamax(search_depth - 1, -alpha -1, -alpha);
            assert(score != -INF);

            // go full depth if score beat alpha
            fullsearch = score > alpha && reduction != 1;

            // go deeper on the full search in case the beats by a margin.
            // Recall that search_depth is the new depth based on the extensions.
            bool deeper = score> best_eval + 70 + 12 * (search_depth - reduction);

            search_depth += deeper;
        }

        // Full depth null window
        if (fullsearch)
        {
            score = -Negamax(search_depth-1, -alpha -1, -alpha);
            assert(score != -INF);
        }

        // PVS
        if (is_pv && (first_move || (score > alpha && score < beta)))
        {
            score = -Negamax(search_depth-1, -beta, -alpha);
            assert(score != -INF);
        }

        undo_move();

        uint64_t end = m_infos.nodes;
        if (is_root)
        {
            m_root_refutation_time[m.raw()] += end - begin;
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
                m_history.update_capture_hist(ss(), captures, m, depth );
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
        info.eval = absolute_eval(best_eval, pos.side_to_move());
        info.nodes_searched = m_infos.nodes;
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
        bestMove = local_best;

    //std::cout << best_valid << " " << local_best << " " << best_eval << " " << evaluate() << std::endl;

    tt_bound_t bound;
    if (best_eval <= alpha_org)
        bound = UPPER;
    else if (best_eval >= beta)
        bound = LOWER;
    else
        bound = EXACT;

    if (best_valid)
        g_tt.store(pos.hash(), depth, store_tt_score(best_eval, ply()), bound, local_best);

    assert(best_eval > -INF && best_eval < INF);
    return best_eval;
}

inline int SearchThread::QSearch(int alpha, int beta)
{
   // std::cout << "Qsearch" << std::endl;
    if (m_thread_id == 0 && m_infos.nodes % 4096 == 0)
    {
        m_tm.update_time();
    }

    m_infos.nodes++;

    bool is_pv = beta - alpha > 1;

    const Position&  pos = m_positions.last();

    //std::cout << positions().back() << evaluate() << " " << alpha << " " << beta  << std::endl;

    if (ply() >= MAX_PLY)
        return evaluate();

    if (is_draw())
        return 0;

    const MoveList moves = gen_legal(pos);
    if (moves.empty())
    {
        if (pos.checkers(pos.side_to_move()))
        {
            return mated_in(ply());
        }
        return 0;
    }

    auto tt_hit = g_tt.probe(pos.hash());
    if (tt_hit)
    {
        do_move<false>(tt_hit->m_move);
        if (is_draw())
            tt_hit = std::nullopt;
        undo_move<false>();

    }
    if (!is_pv && tt_hit)
    {
        const tt_entry_t& e     = *tt_hit;
        const int         score = read_tt_score(e.m_score, ply());
        if (score <= -INF  || score >= INF) std::cout << e.m_score << " " << score << std::endl;
        assert(score > -INF && score < INF);
        if (e.m_bound == EXACT)
            return score;
        if (e.m_bound == LOWER && score >= alpha)
            return score;
        if (e.m_bound == UPPER && score <= beta)
            return score;
    }

    const int stand_pat = evaluate();
    ss().eval = stand_pat;

    //assert(beta > -INF && beta < INF);
    if (stand_pat >= beta)
        return beta;
    if (stand_pat > alpha)
        alpha = stand_pat;

    //std::cout << "qsearch move loop "  << std::endl;
    MoveList tactical = filter_tactical(pos, moves);
    //std::ranges::for_each(tactical, [&](auto m) {std::cout << m.move << std::endl;});

    score_moves(ss(), tactical, tt_hit ? tt_hit->m_move : Move::none(), m_history, ss());
    tactical.sort();

    int best_eval = stand_pat;
    for (auto [m, s] : tactical)
    {
        //std::cout << m << std::endl;
        if (!is_pv && pos.is_occupied(m.to_sq()) && ((s < -5'000'000) ||  pos.piece_at(m.to_sq()).piece_value() + 2*s + best_eval < alpha) )// see pruning on captures, we don't want to look at hopeless captures
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
            best_eval = score;
        if (best_eval > alpha)
            alpha = best_eval;
        if (alpha >= beta)
            break;
    }
    assert(best_eval > -INF && best_eval < INF);
    return best_eval;
}

struct SearchThreadHandler
{
    std::vector<std::unique_ptr<SearchThread>> threads{};
    std::vector<std::jthread>                  workers{};
    TimeManager                                m_tm{};

    void set(const size_t numThreads, const TimeManager& tm, const Position& pos, const std::span<Move> moves)
    {
        threads.clear();
        threads.reserve(numThreads);
        workers.clear();
        workers.reserve(threads.size());
        m_tm = tm;
        for (size_t i = 0; i < numThreads; i++)
        {
            threads.push_back(std::make_unique<SearchThread>(i, m_tm, pos, moves));
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
            move_votes[t->bestMove.raw()]++;
        }

        const auto it =
            std::ranges::max_element(move_votes, [](const auto& a, const auto& b) { return a.second < b.second; });

        return it != move_votes.end() ? Move{it->first} : Move{};
    }

    void stop_all()
    {
        m_tm.stop();
    }
};

#endif // SEARCHER_H
