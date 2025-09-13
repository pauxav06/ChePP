#ifndef TIME_MANAGER_H
#define TIME_MANAGER_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cmath>
#include <iterator>
#include <ranges>

template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buffer(capacity), capacity(capacity), head(0), size_(0) {}

    void push(const T& value) {
        buffer[head] = value;
        head = (head + 1) % capacity;
        if (size_ < capacity) size_++;
    }

    const T& operator[](size_t i) const {
        size_t index = (head + capacity - size_ + i) % capacity;
        return buffer[index];
    }

    size_t size() const { return size_; }
    bool full() const { return size_ == capacity; }

    struct BufferIterator {
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const T*;
        using reference         = const T&;

        const RingBuffer* rb{};
        size_t pos{};

        reference operator*() const { return (*rb)[pos]; }
        pointer operator->() const { return &(*rb)[pos]; }

        BufferIterator& operator++() { ++pos; return *this; }
        BufferIterator operator++(int) { auto tmp = *this; ++*this; return tmp; }
        BufferIterator& operator--() { --pos; return *this; }
        BufferIterator operator--(int) { auto tmp = *this; --*this; return tmp; }

        BufferIterator& operator+=(difference_type n) { pos += n; return *this; }
        BufferIterator& operator-=(difference_type n) { pos -= n; return *this; }

        reference operator[](difference_type n) const { return *(*this + n); }

        friend BufferIterator operator+(BufferIterator it, difference_type n) { it += n; return it; }
        friend BufferIterator operator+(difference_type n, BufferIterator it) { return it + n; }
        friend BufferIterator operator-(BufferIterator it, difference_type n) { it -= n; return it; }
        friend difference_type operator-(BufferIterator a, BufferIterator b) { return difference_type(a.pos) - difference_type(b.pos); }

        friend bool operator==(const BufferIterator& a, const BufferIterator& b) { return a.pos == b.pos; }
        friend bool operator!=(const BufferIterator& a, const BufferIterator& b) { return !(a == b); }
        friend bool operator<(const BufferIterator& a, const BufferIterator& b) { return a.pos < b.pos; }
        friend bool operator>(const BufferIterator& a, const BufferIterator& b) { return b < a; }
        friend bool operator<=(const BufferIterator& a, const BufferIterator& b) { return !(b < a); }
        friend bool operator>=(const BufferIterator& a, const BufferIterator& b) { return !(a < b); }
    };

    auto begin() const { return BufferIterator{this, 0}; }
    auto end()   const { return BufferIterator{this, size_}; }

private:
    std::vector<T> buffer;
    size_t capacity, head, size_;
};


struct TimeManager {

    struct Constraints {
        int move_time{-1};
        EnumArray<Color, int> time{-1, -1};
        EnumArray<Color, int> inc{-1, -1};
        int moves_to_go{-1};
        int depth = 99;
    };

    struct Params {
        int min_time{50};
        int max_time{60 * 1000};
        int safety_margin{200};
        int sampling_depth{10};
        double winning_factor{0.8};
        double losing_factor{1.2};
        double unstable_factor{1.3};
        double stable_factor{0.8};
    };

    struct InitInfo {
        Color side{NO_COLOR};
        int moves_played{0};
        std::vector<int> evaluations{};
    };

    struct UpdateInfo {
        int eval{0};
        int second_move_delta{0};
        uint64_t nodes_searched{0};
    };

    TimeManager() = default;

    explicit TimeManager(const Params& params, const InitInfo& info, const Constraints& constraints)
        : params(params), init_info(info), constraints(constraints), update_infos(params.sampling_depth) {
        compute_base_time();
    }

    void start() {
        start_time = std::chrono::steady_clock::now();
        m_stop_flag = false;
    }

    bool should_stop() const
    {
        return m_stop_flag;
    }

    void send_update_info(const UpdateInfo& info)
    {
        update_infos.push(info);
        adjust_time();
    }

    void update_depth(const int depth)
    {
        std::cout << adjusted_time_ms << " " << depth << std::endl;
        if (depth > 0 && constraints.depth > 0 && depth > constraints.depth) {
            m_stop_flag = true;
        }
    }

    void update_time() {

        if (m_max_time_ms > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= adjusted_time_ms) {
                m_stop_flag = true;
            }
        }
    }

    void stop() { m_stop_flag = true; }

private:

    int estimate_moves_to_go() const {
        int base_moves = constraints.moves_to_go > 0 ? constraints.moves_to_go : 35;

        const auto& evals = init_info.evaluations;
        bool winning = init_info.evaluations.back() > 150;
        bool losing = init_info.evaluations.back() < -150;

        int mean_diff = 0;
        if (evals.size() >= 2) {
            auto diffs = evals
                       | std::views::adjacent<2>
                        | std::views::transform([&](auto&& e) { return std::abs(std::get<0>(e) - std::get<1>(e)); });

            mean_diff = std::accumulate(diffs.begin(), diffs.end(), 0)
                        / static_cast<int>(evals.size() - 1);
        }
        mean_diff = std::clamp(mean_diff, -100, 100);


        if (winning) base_moves = std::max(10, base_moves / (100 + mean_diff) / 100); // we are crushing
        else if (losing) base_moves = std::max(10, base_moves / (100 - mean_diff) / 100); // we are getting crushed

        if (!init_info.evaluations.empty()) {
            const double mean = std::accumulate(init_info.evaluations.begin(),
                                          init_info.evaluations.end(), 0.0) /
                          init_info.evaluations.size();
            const double sq_sum = std::inner_product(init_info.evaluations.begin(),
                                               init_info.evaluations.end(),
                                               init_info.evaluations.begin(),
                                               0.0);
            const double variance = (sq_sum / init_info.evaluations.size()) - (mean * mean);

            if (variance > 50) base_moves = static_cast<int>(base_moves * 1.5); // unstable -> longer
            if (std::abs(mean) < 30 && variance < 10) base_moves = static_cast<int>(base_moves * 1.5); // dead draw -> very long
        }

        return base_moves;
    }


    void compute_base_time() {
        if (constraints.move_time > 0) {
            m_max_time_ms = constraints.move_time;
            adjusted_time_ms = m_max_time_ms;
            return;
        }
        const int time_left = constraints.time[init_info.side];
        const int inc = constraints.inc[init_info.side];
        if (time_left < 0) {
            m_max_time_ms = params.max_time;
            adjusted_time_ms = m_max_time_ms;
            return;
        }
        const int moves_to_go = constraints.moves_to_go > 0 ? constraints.moves_to_go : 35;
        m_max_time_ms = std::max(params.min_time, (time_left / moves_to_go) + inc);
        m_max_time_ms = std::max(params.min_time, m_max_time_ms - params.safety_margin);
        m_max_time_ms = std::min(m_max_time_ms, params.max_time);
        adjusted_time_ms = m_max_time_ms;
    }

    void adjust_time() {
        if (update_infos.size() < 2) return;

        int last_eval = update_infos[update_infos.size() - 1].eval;
        int prev_eval = update_infos[update_infos.size() - 2].eval;

        auto evals = update_infos | std::views::transform(&UpdateInfo::eval);
        int sum  = std::ranges::fold_left(evals, 0.0, std::plus<>());
        int mean = sum / update_infos.size();

        int sq_sum = std::ranges::fold_left(evals | std::views::transform([](const int e){ return e * e; }),0, std::plus<>());

        auto variance = (sq_sum / update_infos.size()) - (mean * mean);


        bool winning = last_eval > 100;
        bool losing  = last_eval < -100;
        bool unstable = variance > 50;
        bool stable = variance < 10;

        double factor = 1.0;
        if (winning) factor *= params.winning_factor;
        if (losing)  factor *= params.losing_factor;
        if (unstable) factor *= params.unstable_factor * (1.0 + (variance + 50) / 100.0);
        if (stable) factor *= params.stable_factor;

        adjusted_time_ms = std::clamp(static_cast<int>(m_max_time_ms * factor), params.min_time, params.max_time);
    }

    Params params{};
    InitInfo init_info{};
    Constraints constraints{};
    RingBuffer<UpdateInfo> update_infos{0};
    std::chrono::steady_clock::time_point start_time{};
    int m_max_time_ms{-1};
    int adjusted_time_ms{-1};
    bool m_stop_flag{false};
};

#endif // TIME_MANAGER_H
