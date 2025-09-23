#ifndef TIME_MANAGER_H
#define TIME_MANAGER_H

#include "types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

struct TimeManager {

    struct UCIConstraints {
        int move_time{-1};
        EnumArray<int, Color> time{-1, -1};
        EnumArray<int, Color> inc{-1, -1};
        int moves_to_go{-1};
        int depth = 99;
    };

    struct Params {
        int baseline_moves_to_go{30};
        int min_time{50};
        int max_time{60 * 1000 * 60};
        int safety_margin{200};
        float instabiliy_factor{0.05f};
        float stability_factor{0.01f};
        float killer_factor{1.5f};
    };

    struct InitInfo {
        Color side{NO_COLOR};
        int moves_played{0};
        int static_eval{};
    };

    struct UpdateInfo {
        int eval{0};
        int second_move_delta{0};
        bool changed{false};
        uint64_t nodes_searched{0};
    };

    struct State
    {
        std::chrono::steady_clock::time_point start_time{};
        int max_time_ms{-1};
        int adjusted_time_ms{-1};
        bool stop_flag{false};
        int same_move_streak{0};
        int n_changes{0};
        int score{0};
    };

    TimeManager() = default;

    explicit TimeManager(const Params& params, const InitInfo& info, const UCIConstraints& constraints)
        : m_params(params), m_init_info(info), m_constraints(constraints) {
        compute_base_time();
    }

    void start() {
        m_state.start_time = std::chrono::steady_clock::now();
        m_state.stop_flag = false;
    }

    [[nodiscard]] bool should_stop() const
    {
        return m_state.stop_flag;
    }


    // called by iterative deepening
    void adjust_time(const UpdateInfo& info) {
        return;
        if (info.changed)
        {
            m_state.same_move_streak = 0;
            m_state.n_changes++;
        }
        else m_state.same_move_streak++;
        m_state.adjusted_time_ms *= std::pow(1.0f - m_params.stability_factor, m_state.same_move_streak);
        m_state.adjusted_time_ms = clamp_time(m_state.adjusted_time_ms);
    }

    void new_killer()
    {
        return;
        m_state.adjusted_time_ms *= m_params.killer_factor;
        m_state.adjusted_time_ms = clamp_time(m_state.adjusted_time_ms);
    }

    void update_depth(const int depth)
    {
        if (depth > 0 && m_constraints.depth > 0 && depth > m_constraints.depth) {
            m_state.stop_flag = true;
        }
    }

    void update_time() {
        if (m_state.max_time_ms > 0) {
            auto elapsed = std::chrono::steady_clock::now() - m_state.start_time;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= m_state.adjusted_time_ms) {
                m_state.stop_flag = true;
            }
        }
    }

    void stop() { m_state.stop_flag = true; }

private:

    [[nodiscard]] int estimate_moves_to_go() const {
        assert(m_constraints.moves_to_go < 0);
        return 30;
        const int baseline = m_params.baseline_moves_to_go;
        int estimate = baseline;
        estimate -= std::min(std::abs(m_init_info.static_eval / 100), 10); // if high eval we will probably finish sooner
        estimate += (m_init_info.moves_played > 75 ? m_init_info.moves_played / 30 : 0); // very long drawish games
        return estimate;
    }


    void compute_base_time() {
        // first we check if we have a fixed time to search
        if (m_constraints.move_time > 0) {
            m_state.max_time_ms = clamp_time(m_constraints.move_time - m_params.max_time);
            m_state.adjusted_time_ms = m_constraints.move_time;
            return;
        }
        const int time_left = m_constraints.time[m_init_info.side];
        const int inc = m_constraints.inc[m_init_info.side];
        if (time_left < 0) { // if we get no time we assume we can go on forever
            m_state.max_time_ms = m_params.max_time;
            m_state.adjusted_time_ms = m_params.max_time;
            return;
        }
        const int moves_to_go = m_constraints.moves_to_go > 0 ? m_constraints.moves_to_go : estimate_moves_to_go();
        m_state.max_time_ms = std::max(m_params.min_time, (time_left / moves_to_go) + inc);
        m_state.max_time_ms = clamp_time(m_state.max_time_ms);
        m_state.adjusted_time_ms = m_state.max_time_ms;
    }

    [[nodiscard]] int clamp_time(int time) const
    {
        time  = std::max(m_params.min_time, time - m_params.safety_margin);
        time = std::min(time, m_params.max_time);
        return time;
    }

    Params m_params{};
    InitInfo m_init_info{};
    UCIConstraints m_constraints{};
    State m_state{};

};

#endif // TIME_MANAGER_H
