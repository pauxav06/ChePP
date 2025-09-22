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

inline std::function<int(bool, int)> default_lmp = [](const bool improving, int d)
{
    d = std::clamp(d, 0, 8);
    return improving ?
        static_cast<int>(4 + 4 * d * d / 4.5) :
        static_cast<int>(2.5 + 2 * d * d / 4.5);
};


struct SearchThread
{
    // can be overridden by UCI default params, and themself overridden by user
    struct Parameters
    {
        int  n_pv{1};
        bool use_syzygy{false};

        int aspiration_window_activation_depth{8};
        int aspiration_window_default_value{50};
        int aspiration_window_multiplicative_factor{2};

        std::function<int(bool, int)>      lmp{default_lmp};
        std::function<int(bool, int, int)> lmr{default_lmr};

        Scorer<ScorerType::Search>::Params search_scorer_params{};
        Scorer<ScorerType::Root>::Params root_scorer_params{};
        Scorer<ScorerType::QSearch>::Params qsearch_scorer_params{};

        int max_history = 16384;
        int max_counter_history = 16384;

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
        MoveList line{};
    };

    using PvLines = std::vector<PvLine>;

    explicit SearchThread(const Parameters& parameters, const int id, TimeManager* tm, TT* tt, const Positions& pos)
        : m_thread_id(id), m_parameters(parameters), m_tm(tm), m_tt(tt), m_search_stack(pos)
    {
        init_cache();
    }
    //control
    int                   m_thread_id;
    Parameters            m_parameters;
    //shared states
    TimeManager*          m_tm;
    TT*                   m_tt;
    //thread local states
    Cache                 m_cache;
    Statistics            m_statistics;
    SearchStack           m_search_stack;
    PvLine                m_pv_line{};


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
        //assert(!ss().position->in_check(ss().position->side_to_move()));
        //assert(!is_draw());

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


    MoveList get_pv(const Position& startpos, const Move first_move) const
    {
        Positions positions{};
        positions.set_pos(startpos);
        Move move = first_move;
        MoveList moves{};
        while (true)
        {
            if (!positions.last().is_valid(move)) break;
            if (positions.is_repetition() || positions.last().halfmove_clock() >= 100) break;
            if (positions.ply() >= MAX_PLY)break;
            moves.push_back(move);
            positions.do_move(move);
            auto tt_hit = m_tt->probe(positions.last().hash());
            if (!tt_hit) break;
            if (moves.size() == MAX_MOVES) break;
            move = tt_hit->move;
        }
        return moves;
    }


    [[nodiscard]] std::string format_pv_line(const MoveList& pv_line) const
    {
        std::ostringstream oss;
        for (const auto m : pv_line)
        {
            oss << m << " ";
        }
        return oss.str();
    }

    auto make_replacement_policy() const
    {
        return [this] (const TT::Entry& old, const TT::Entry& candidate)
        {

            bool replace = !old.hash || (old.generation != candidate.generation) || old.depth <= candidate.depth;
            if (!replace) return false;
            if (candidate.move || old.hash != candidate.hash) return true;
            if (candidate.bound == TT::EXACT || candidate.hash != old.hash || candidate.depth + 4 > old.depth) return true;
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
    int  Negamax(int depth, int alpha, int beta, bool cutnode);
    int  QSearch(int alpha, int beta);
};

inline void SearchThread::IterativeDeepening()
{
    int prev_eval = evaluate();
    m_statistics.t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < m_parameters.n_pv; ++i)
    {
        for (int depth = 1; m_tm->update_depth(depth), !m_tm->should_stop(); ++depth)
        {
            const int eval = AspirationWindow(depth, prev_eval);
            assert(ss().position);

            assert(eval > -INF && eval < INF);

            if (!m_tm->should_stop()) // aspiration window got cancelled, we discard the result
            {
                prev_eval = eval;

                if (m_thread_id == 0)
                {
                    std::string score;
                    if (eval >= MATE_IN_MAX_PLY)
                    {
                        score.append("mate ");
                        score.append(std::to_string((MATE - eval) / 2));
                    }
                    else if (eval <= MATED_IN_MAX_PLY)
                    {
                        score.append("mate ");
                        score.append(std::to_string((MATED - eval) / 2));
                    }
                    else
                    {
                        score.append("cp ");
                        score.append(std::to_string(eval));
                    }

                    auto t_now = std::chrono::high_resolution_clock::now();
                    auto time_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - m_statistics.t_start);
                    time_since_start = std::max(time_since_start, std::chrono::milliseconds(1));
                    int nps = m_statistics.nodes / time_since_start.count();
                    auto pv = get_pv(ss().position(), ss().best_move);
                    std::string uci_output = std::format("info score {} depth {} nodes {} nps {} tb_hits {} pv {}",
                        score, depth, m_statistics.nodes, nps, m_statistics.tb_hits, format_pv_line(pv));
                    std::cout << uci_output << std::endl;

                    TimeManager::UpdateInfo update_info;
                    update_info.eval = eval;
                    m_tm->adjust_time(update_info);
                }
            }
        }
    }
}

inline int SearchThread::AspirationWindow(const int depth, const int prev_eval)
{
    if (depth < m_parameters.aspiration_window_activation_depth)
    {
        return Negamax(depth, -INF_SCORE, INF_SCORE, false);
    }

    int window = m_parameters.aspiration_window_default_value;
    int alpha  = prev_eval - window;
    int beta   = prev_eval + window;

    auto eval = Negamax(depth, alpha, beta, false);

    while (eval <= alpha || eval >= beta)
    {
        if (m_tm->should_stop())
            break;

        window *= m_parameters.aspiration_window_multiplicative_factor;
        alpha = std::clamp(eval - window, -INF_SCORE, INF_SCORE);
        beta  = std::clamp(eval + window, -INF_SCORE, INF_SCORE);

        eval = Negamax(depth, alpha, beta, false);
    }
    return eval;
}

inline int SearchThread::Negamax(int depth, int alpha, int beta, bool cutnode)
{
    //assert(depth >= 0);

    // quiescence search supposed to prevent horizon effect
    if (depth <= 0)
        return QSearch(alpha, beta);


    if (m_thread_id == 0 && m_statistics.nodes % 4096 == 0)
    {
        m_tm->update_time();
    }

    const int  alpha_org = alpha;
    const bool is_pv = beta - alpha > 1;
    const bool is_root   = ply() == 0;
    const bool in_check  = ss().position->in_check(ss().position->side_to_move());
    bool improving = false;
    int eval = 0;

    // increase depth if we are in check
    if (in_check)
    {
        depth++;
    }



    m_statistics.nodes++;


    if (!is_root)
    {
        if (ply() >= MAX_PLY)
        {
            return evaluate();
        }
        if (is_draw())
        {
            return 0;
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

    // Probe the TT to see if we have a candidate score
    auto tt_hit = ss().excluded ? std::nullopt : m_tt->probe(ss().position->hash());
    int tt_score = tt_hit ? TT::read_score(tt_hit->score, ply()) : 0;
    Move tt_move = tt_hit ? tt_hit->move : Move::none();
    if (tt_hit && tt_move) tt_hit = causes_draw(tt_hit->move) ? std::nullopt : tt_hit;
    if (!is_pv && tt_hit && tt_hit->depth >= depth && tt_move)
    {
        const int score = TT::read_score(tt_score, ply());
        if (tt_hit->bound == TT::Bound::EXACT || (tt_hit->bound == TT::Bound::LOWER && score >= alpha) || (tt_hit->bound == TT::Bound::UPPER && score <= beta))
        {
            m_statistics.tt_hits++;
            return tt_score;
        }

    }



    ss().static_eval = eval = tt_hit ? tt_hit->static_eval : evaluate();
    improving = !in_check && (ply() >= 2 && ss().static_eval > ss().prev->prev->static_eval);


    if (in_check || ss().excluded)
    {
        ss().static_eval = eval = 0;
        improving = false;
    }

    if (!is_pv && !in_check && !is_root && !ss().excluded)
    {
        if (tt_hit)
        {
            eval = tt_score;
        }

        if (ply() >= 2 && depth < 9 && eval >= beta
            && ss().position->occupancy().popcount() > 4 //disable when scores get near mate
            && eval - ((depth - improving) * 77) - ss().prev->prev->static_eval/400 >= beta)
        {
            return eval;
        }


        if (cutnode && ss().static_eval > (beta - 76 * improving)
            && ss().position->occupancy(KNIGHT, BISHOP, ROOK, QUEEN).popcount() >= 1
            && depth >= 3 && ss().position->move() != Move::null()
            && (!tt_hit || tt_hit->bound == TT::LOWER || eval >= beta))
        {
            int reduction = 3 + depth / 3;

            do_move(Move::null());
            int score = -Negamax(depth - reduction, -beta, -beta + 1, !cutnode);
            undo_move();

            if (m_tm->should_stop())
            {
                return 0;
            }

            if (score >= beta && score < MATE_IN_MAX_PLY)
            {
                return score;
            }
        }

        int rbeta = std::min(beta + 100, MATE - MAX_PLY - 1);
        if (depth >= 3 && abs(beta) < MATE_IN_MAX_PLY &&
            (!tt_hit || eval >= rbeta || tt_hit->depth < depth - 3))
        {

            int score = 0;
            auto list = gen_legal(ss().position())
            | std::views::filter(make_tactical_predicate(ss().position()));

            Scorer<ScorerType::QSearch> scorer(m_parameters.qsearch_scorer_params, ss(), tt_move);

            for (const auto [m, s] : ScoredMoveStream(list, scorer()))
            {
                if (s < scorer.m_params.good_capture_bonus) continue;
                if (m == ss().excluded) continue;

                do_move(m);
                score = -QSearch(-rbeta,-rbeta  + 1);

                if (score >= rbeta)
                {
                    score  = -Negamax(depth - 4, -rbeta, -rbeta + 1, !cutnode);
                }

                undo_move();

                if (score >= rbeta)
                {
                    m_tt->store(make_replacement_policy(), ss().position->hash(), depth - 3, score, TT::UPPER, m, ss().static_eval);
                    return score;
                }
            }
        }

        if (false && eval - 63 + 182 * depth <= alpha)
        {
            return QSearch(alpha, beta);
        }
    }

    if (cutnode && depth >= 7 && tt_move == Move::none())
    {
        depth--;
    }

    int best_score= -INF;
    int move_count = 0;
    int score = -INF;
    Move best_move = Move::none();


    MoveList moves = gen_legal(ss().position());

    if (moves.empty())
    {
        return in_check ? mated_in(ply()) : 0;
    }

    bool skip_quiets = false;

    auto get_move_stream = [&] (const MoveList& move_list)
    {
        if (false && is_root && depth > 7)
            return ScoredMoveStream(move_list, Scorer<ScorerType::Root>(m_parameters.root_scorer_params, ss(), tt_move)());
        return ScoredMoveStream(move_list, Scorer<ScorerType::Search>(m_parameters.search_scorer_params, ss(), tt_move)());
    };

    MoveList quiet_list;

    for (const auto [m, s] : get_move_stream(moves))
    {
        if (m == ss().excluded) continue;

        bool is_quiet = ss().position->is_quiet(m);
        int extension = 0;


        bool refutation_move = (ss().killer1 == m || ss().killer2 == m);

        if (is_quiet && skip_quiets) continue;

        if (!is_root && best_score > MATED)
        {

            int lmr_depth = m_cache.lmr.at(is_quiet).at(std::min(depth, MAX_PLY - 1)).at(std::min(move_count, MAX_MOVES - 1));


            if (is_quiet)
            {
                if (false && !in_check && !is_pv && depth <= 7 && quiet_list.size() > m_cache.lmp.at(improving).at(depth))
                {
                    skip_quiets = true;
                    continue;
                }

                if (false && lmr_depth < 3 && !refutation_move && s < -4000 * depth)
                {
                    continue;
                }

                if (false && lmr_depth <= 6 && !in_check && eval + 217 + 71 * depth <= alpha)
                {
                    skip_quiets = true;
                }

            }
            else
            {
                if (false && depth <= 6 && ss().position->see(m) < -15 * depth * depth)
                {
                    continue;
                }
            }
        }



        if (!is_root && depth >= (6 + is_pv) && (m == tt_move)
            && tt_hit->bound == TT::LOWER && abs(tt_score) < MATE
            && tt_hit->depth >= depth - 3)
        {
            int singular_beta = tt_score - depth;
            int singular_depth = (depth - 1) / 2;

            ss().excluded = tt_move;
            int singular_score = Negamax(singular_depth, singular_beta - 1,  singular_beta, cutnode);
            ss().excluded = Move::none();

            if (singular_score < singular_beta && ss().extensions < 5) {
                extension = 1;
                ss().extensions++;
                if (!is_pv && singular_score < singular_beta - 20 && ss().double_extensions < 2){
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

        m_statistics.nodes++;
        move_count++;

        if (is_root && depth == 1 && move_count == 1)
        {
            ss().best_move = m;
        }

        if (is_quiet)
        {
            quiet_list.push_back(m);
        }

        bool do_full_search = !is_root || move_count > 1;


        if (!in_check && do_full_search && depth >= 3 &&
            move_count > (2 + 2* is_pv))
        {
            int reduction = m_cache.lmr.at(is_quiet).at(std::min(depth, MAX_PLY - 1)).at(std::min(move_count, MAX_MOVES - 1));

            reduction += !improving;
            reduction += is_pv;
            reduction += is_quiet;

            reduction -= is_quiet ? refutation_move ? 2 : s/4000 : 0;

            reduction = std::min(depth - 1, std::max(1, reduction));

            score = -Negamax(new_depth - reduction, -alpha - 1, -alpha, true);

            do_full_search = score > alpha && reduction != 1;

            bool deeper = score > best_score + 70 + 12 * (new_depth - reduction);
            new_depth += deeper;
        }

        if (do_full_search)
        {
            score  =-Negamax(new_depth - 1, -alpha - 1, -alpha, !cutnode);
        }

        if (is_pv && (move_count == 1 || (score > alpha && score < beta)))
        {
            score = -Negamax(new_depth - 1, -beta, -alpha, false);
        }

        undo_move();

        auto end = m_statistics.nodes;

        if (is_root)
        {
            if (!ss().refutation_nodes->contains(m)) ss().refutation_nodes->emplace(m, 0);
            ss().refutation_nodes->at(m) += end - start;
        }

        if (m_tm->should_stop() && !is_root)
        {
            return 0;
        }


        if (score > best_score)
        {
            best_score = score;
            best_move = m;
            if (score > alpha)
            {
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
                    }
                    break;
                }
            }
        }

        if (m_tm->should_stop() && is_root && ss().best_move)
        {
            break;
        }

    }

    {
        auto get_bonus_fn = [&] (const Move move)
        {
            return [=](const MoveScoreT b) -> MoveScoreT {
                if (move == best_move) return std::clamp(b + depth * depth, -16000, 16000);
                return std::clamp(b - b / 5, -16000, 16000);
            };
        };

        for (const auto m : quiet_list)
        {
            ss().history.apply_bonus(best_move, get_bonus_fn(m));
            if (ply() > 1) ss().continuation_history.apply_bonus(ss().position(), m,  get_bonus_fn(m));
            if (ply() > 2) ss().prev->continuation_history.apply_bonus(ss().position(), m,  get_bonus_fn(m));
        }
    }

    TT::Bound bound = (best_score <= alpha_org) ? TT::UPPER : (best_score >= beta) ? TT::LOWER : TT::EXACT;
    if (!ss().excluded)
        m_tt->store(make_replacement_policy() ,ss().position->hash(), depth, TT::store_score(best_score, ply()), bound, best_move, ss().static_eval);

    if (alpha != alpha_org)
    {
        ss().best_move = best_move;
    }

    return best_score;
}

inline int SearchThread::QSearch(int alpha, int beta)
{
    //assert(beta > -INF && beta < INF);

    if (m_thread_id == 0 && m_statistics.nodes % 4096 == 0) m_tm->update_time();

    const bool is_pv = beta - alpha > 1;
    const bool in_check = ss().position->in_check(ss().position->side_to_move());

    if (ply() >= MAX_PLY)
        return evaluate();

    if (is_draw())
        return 0;



    const int stand_pat = evaluate();
    ss().static_eval           = stand_pat;

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    auto tt_hit = m_tt->probe(ss().position->hash());
    //if (tt_hit) tt_hit = causes_draw(tt_hit->move) ? std::nullopt : tt_hit;
    const Move tt_move = tt_hit ? tt_hit->move : Move::none();
    const int tt_score = tt_hit ? TT::read_score(tt_hit->score, ply()) : 0;
    if (!is_pv && tt_hit)
    {
        assert(tt_score > -INF && tt_score < INF);
        if (tt_hit->bound == TT::EXACT || (tt_hit->bound == TT::LOWER && tt_score >= alpha) || (tt_hit->bound == TT::UPPER && tt_score <= beta)) {
            return tt_score;
        }
    }


    auto moves = gen_legal(ss().position());
    if (moves.empty()) return in_check ? mated_in(ply()) : 0;
    auto tactical = moves | std::views::filter(make_tactical_predicate(ss().position()));
    Scorer<ScorerType::QSearch> scorer(m_parameters.qsearch_scorer_params, ss(), tt_move);

    int  best_eval = stand_pat;
    Move best_move = Move::none();
    int move_count = 0;
    int score = -INF;

    for (const auto [m, s] : ScoredMoveStream(tactical, scorer()))
    {
        if (s < scorer.m_params.good_capture_bonus && move_count > 1)
        {
            continue;
        }

        do_move(m);

        m_statistics.nodes++;
        move_count++;

        score = -QSearch(-beta, -alpha);

        undo_move();

        if (m_tm->should_stop())
        {
            return 0;
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
    assert(best_eval > -INF && best_eval < INF);
    if (best_move != Move::none() && best_move != Move::null())
    {
        TT::Bound bound = (best_eval >= beta) ? TT::LOWER : TT::UPPER;
        //m_tt->store(make_replacement_policy(), ss().position->hash(), 0, TT::store_score(best_eval, ply()), bound, best_move, stand_pat);
    }

    return best_eval;
}

struct SearchThreadHandler
{
    std::vector<std::unique_ptr<SearchThread>> threads{};
    std::vector<std::jthread>                  workers{};

    TimeManager                                m_tm{};
    TT*                                        m_tt;

    void set(
        const size_t numThreads,
        const SearchThread::Parameters& params,
        const TimeManager::Params& tm_params,
        const TimeManager::UCIConstraints& tm_constraints,
        TT* tt,
        const Positions& pos)
    {
        threads.clear();
        threads.reserve(numThreads);
        workers.clear();
        workers.reserve(threads.size());
        const TimeManager::InitInfo tm_init{
            .side = pos.last().side_to_move(),
            .moves_played =(pos.last().full_move_clock() / 2),
            .static_eval = Accumulator(pos.last()).evaluate(pos.last().side_to_move())
        };
        m_tm = TimeManager{tm_params, tm_init, tm_constraints};
        m_tt = tt;
        for (size_t i = 0; i < numThreads; i++)
        {
            threads.emplace_back(std::make_unique<SearchThread>(params, i, &m_tm, tt, pos));
        }
    }

    void start()
    {
        workers.clear();
        m_tt->new_generation();
        m_tm.start();

        for (const auto& thread : threads)
        {
            auto t = thread.get();
            workers.emplace_back([t]() { t->IterativeDeepening(); });
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
            move_votes[t->ss().best_move.raw()]++;
        }

        const auto it =
            std::ranges::max_element(move_votes, [](const auto& a, const auto& b) { return a.second < b.second; });

        return it != move_votes.end() ? Move{it->first} : Move{};
    }

    void stop_all() { m_tm.stop(); }
};

#endif // SEARCHER_H
