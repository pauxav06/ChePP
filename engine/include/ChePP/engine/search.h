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
        : m_thread_id(id), m_tm(tm), m_positions(pos, moves), m_accumulators(m_positions.last()), m_ss(MAX_PLY + 1)
    {}

    int          m_thread_id;
    TimeManager& m_tm;

    Positions                          m_positions;
    Accumulators                       m_accumulators;
    SearchStack                        m_ss;

    SearchInfos    m_infos{};
    HistoryManager m_history{};

    Move bestMove;


    [[nodiscard]] int ply() const { return static_cast<int>(m_positions.ply()); }
    [[nodiscard]] SearchStack::Node& ss() { return m_ss[ply()]; }

    template <bool UpdateNNUE = true>
    void do_move(const Move move)
    {
        m_positions.do_move(move);
        if constexpr (UpdateNNUE) m_accumulators.do_move(m_positions[ply() - 1], m_positions.last());
        ss().move = move;
    }

    template <bool UpdateNNUE = true>
    void undo_move()
    {
        ss().move = Move::none();
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

    [[nodiscard]] bool is_repetition() const { return m_positions.is_repetition(); }

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

inline const std::array<std::array<int, 256>, MAX_PLY>& lmr_table()
{
    static std::array<std::array<int, 256>, MAX_PLY> g_lmr_table = []()
    {
        std::array<std::array<int, 256>, MAX_PLY> lmr{};
        for (int d = 1; d < MAX_PLY; ++d)
        {
            for (int m = 1; m < 256; ++m)
            {
                lmr[d][m] = static_cast<int>(0.99 + std::log(d) * std::log(d) / 3.14);
            }
        }
        return lmr;
    }();
    return g_lmr_table;
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

    ret.depth = depth;
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

    if (depth <= 5) {
        alpha = -INF;
        beta  = +INF;
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
        alpha = eval - window;
        beta  = eval + window;

        eval = Negamax(depth, alpha, beta);
    }

    stats.update(eval - prev_eval);

    return eval;
}


inline auto store_tt_score(const int score, const int ply)
{
    if (score >= MATE_IN_MAX_PLY)
        return score + ply;
    if (score <= -MATE_IN_MAX_PLY)
        return score - ply;
    return score;
};

inline auto read_tt_score(const int score, const int ply)
{
    if (score >= MATE_IN_MAX_PLY)
        return score - ply;
    if (score <= -MATE_IN_MAX_PLY)
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
        info.eval = depth;
        m_tm.update_time();
    }
    const Position&        pos = m_positions.last();

    const int  alpha_org = alpha;
    const bool is_root   = ply() == 0;
    const bool in_check  = pos.checkers(pos.side_to_move()).value();


    depth += in_check;

    if (depth <= 0)
        return QSearch(alpha, beta);

    m_infos.nodes++;

    if (!is_root)
    {
        if (is_repetition())
        {
            return 0;
        }

        if (ply() >= MAX_PLY)
        {
            return evaluate();
        }

        // this speeds up mate cases
        // our worse move is to be mated on the spot
        alpha = std::max(alpha, mated_in(static_cast<int>(ply())));
        // their best move is to mate next turn
        beta = std::min(beta, mate_in(ply() + 1));

        if (alpha >= beta)
        {
            return alpha;
        }
    }

    const bool is_pv = beta - alpha > 1;


    auto tt_hit = g_tt.probe(pos.hash());
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

    const int static_eval = tt_hit ? tt_hit->m_score : evaluate();
    ss().eval = static_eval;

    MoveList moves = gen_legal(pos);

    if (moves.empty())
    {
        return in_check ? mated_in(ply()) : 0;
    }

    score_moves(positions(), moves, tt_hit ? tt_hit->m_move : Move::none(), m_history, ss());
    moves.sort();


    if (!is_root && !is_pv && !in_check && static_eval - depth * 100 >= beta)
    {
        return static_eval;
    }

    // null move pruning
    // idea : if we expect we will beat beta, we offer a free move and search at reduced depth
    // if eval comes from tt, is upper bounded and not higher that beta, we cant assume anything on score
    // evaluating is not worth it so we just skip
    // only do it if there are enough pieces to not avoid zugzwang blindness
    if (!is_root && !is_pv && positions().back().move() != Move::null() && !in_check && depth >= 3 && static_eval >= beta &&
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

    if (!is_root && !is_pv && !in_check && depth >= 3 && static_eval >= beta + 150)
    {
        int       prob_beta = beta + 150;
        const int reduction = 3;

        MoveList  tactical  = filter_tactical(pos, gen_legal(pos));
        score_moves(positions(), tactical, tt_hit ? tt_hit->m_move : Move::none(), m_history, ss());
        tactical.sort();

        for (auto [m, s] : tactical)
        {
            if (m == tt_hit->m_move || s < -1000)
            {
                continue;
            }
            do_move(m);

            auto score = -QSearch(-prob_beta, -prob_beta + 1);

            if (score >= prob_beta)
            {
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


    int      best_eval  = -INF_SCORE;
    Move     local_best = Move::none();
    bool     first_move = true;
    int      move_idx   = 0;
    MoveList quiets{};

    for (auto [m, s] : moves)
    {

        bool is_quiet = !pos.is_occupied(m.to_sq()) && m.type_of() != EN_PASSANT && m.type_of() != PROMOTION;
        if (is_quiet)
            quiets.push_back(m);



        if (!is_root && !is_pv && !in_check && best_eval != -INF_SCORE && is_quiet && depth <= FUTILITY_DEPTH_MAX)
        {
            const int margin = futility_margin_for_depth(depth);
            if (static_eval + margin <= alpha)
            {
                move_idx++;
                first_move = false;
                continue;
            }
        }

        if (!is_root && !is_pv && !in_check && best_eval != -INF_SCORE && is_quiet && depth <= 3)
        {
            const int lmp_limit = 3 + depth * depth;
            if (move_idx >= lmp_limit)
            {
                move_idx++;
                first_move = false;
                continue;
            }
        }

        int search_depth = depth;



        do_move(m);

        int score;

        if (depth >= 3 && !in_check && move_idx > 0)
        {
            int reduction = std::min(lmr_table()[depth][move_idx], depth - 1);
            //reduction += is_quiet;
            //reduction -= s / 4000;
            reduction = std::max(reduction, 0);
            search_depth -= reduction;
            search_depth = std::max(search_depth, 2);
        }

        if ((is_root && depth < 7) || first_move || in_check)
        {
            score = -Negamax(search_depth - 1, -beta, -alpha);
        }
        else
        {
            score = -Negamax(search_depth - 1, -alpha - 1, -alpha);

        }
        if (score > alpha && score < beta)
        {
            score = -Negamax(depth - 1, -beta, -alpha);
        }

        undo_move();

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
                m_history.update_cont_hist(positions(), quiets, m, depth, std::min(2, static_cast<int>(ply())));
                m_history.update_hist(positions().back(), quiets, m, depth);
            }
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

    bool best_valid = !m_tm.should_stop() && local_best != Move::none();
    if (is_root && best_valid)
        bestMove = local_best;

    tt_bound_t bound;
    if (best_eval <= alpha_org)
        bound = UPPER;
    else if (best_eval >= beta)
        bound = LOWER;
    else
        bound = EXACT;

    if (best_valid)
        g_tt.store(pos.hash(), depth, store_tt_score(best_eval, ply()), bound, local_best);

    return best_eval;
}

inline int SearchThread::QSearch(int alpha, int beta)
{
    m_infos.nodes++;

    bool is_pv = beta - alpha > 1;

    const Position&  pos = m_positions.last();

    //std::cout << positions().back() << evaluate() << " " << alpha << " " << beta  << std::endl;

    if (ply() >= MAX_PLY)
        return evaluate();

    if (is_repetition())
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
    if (!is_pv && tt_hit)
    {
        const tt_entry_t& e     = *tt_hit;
        const int         score = read_tt_score(e.m_score, ply());
        if (e.m_bound == EXACT)
            return score;
        if (e.m_bound == LOWER && score >= alpha)
            return score;
        if (e.m_bound == UPPER && score <= beta)
            return score;
    }

    const int stand_pat = evaluate();
    ss().eval = stand_pat;

    if (stand_pat >= beta)
        return beta;
    if (stand_pat > alpha)
        alpha = stand_pat;

    //std::cout << "qsearch move loop "  << std::endl;
    MoveList tactical = filter_tactical(pos, moves);
    //std::ranges::for_each(tactical, [&](auto m) {std::cout << m.move << std::endl;});

    score_moves(positions(), tactical, tt_hit ? tt_hit->m_move : Move::none(), m_history, ss());
    tactical.sort();

    int best_eval = stand_pat;
    for (auto [m, s] : tactical)
    {
        //std::cout << m << std::endl;
        if (!is_pv && pos.is_occupied(m.to_sq()) && s < -1000) // see pruning on captures
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

    void start(const std::function<void()>& Cb)
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

        Cb();
        threads.clear();
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

    void stop_all() { m_tm.stop(); }
};

#endif // SEARCHER_H
