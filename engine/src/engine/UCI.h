//
// Created by paul on 9/6/25.
//

#ifndef CHEPP_UCI_H
#define CHEPP_UCI_H
#include "core.h"
#include "search.h"
#include "tm.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chepp {
    using UciCallbacT       = std::function<bool()>;
    inline auto uci_cb_none = []() { return true; };

    class EngineParameter {
      public:
        explicit EngineParameter(const std::string& name, UciCallbacT cb = uci_cb_none)
            : m_name(name), m_cb(std::move(cb)) {
        }
        virtual ~EngineParameter() = default;

        [[nodiscard]] const std::string&
        name() {
            return m_name;
        };
        [[nodiscard]] virtual std::string
        uci_declare() const = 0;
        virtual bool
        parse([[maybe_unused]] const std::string& value) {
            return m_cb();
        };
        [[nodiscard]] virtual std::string
        value_str() const = 0;

      protected:
        std::string m_name;
        UciCallbacT m_cb;
    };

    template <typename T>
    class ValueEngineParameter : public EngineParameter {
      public:
        explicit ValueEngineParameter(const std::string& name, T& underlying, T init, UciCallbacT cb = uci_cb_none)
            : EngineParameter(name, std::move(cb)), m_init(std::move(init)), m_value(underlying) {
            m_value = m_init;
        }

        ~ValueEngineParameter() override = default;

        [[nodiscard]] T
        value() const {
            return m_value;
        }

      protected:
        T  m_init{};
        T& m_value{};
    };

    class EngineParamCheck final : public ValueEngineParameter<bool> {
      public:
        explicit EngineParamCheck(const std::string& name,
                                  bool&              underlying,
                                  const bool         def = false,
                                  UciCallbacT        cb  = uci_cb_none)
            : ValueEngineParameter(name, underlying, def, std::move(cb)) {
        }

        [[nodiscard]] std::string
        uci_declare() const override {
            return fmt::format("option name {} type check default {}", m_name, m_init ? "true" : "false");
        }

        bool
        parse(const std::string& v) override {
            if (v == "true" || v == "1") {
                m_value = true;
                return EngineParameter::parse(v);
            }
            if (v == "false" || v == "0") {
                m_value = false;
                return EngineParameter::parse(v);
            }
            return false;
        }

        [[nodiscard]] std::string
        value_str() const override {
            return m_value ? "true" : "false";
        }
    };

    class EngineParamSpin final : public ValueEngineParameter<int> {
      public:
        EngineParamSpin(const std::string& name,
                        int&               underlying,
                        const int          init,
                        const int          min,
                        const int          max,
                        UciCallbacT        cb = uci_cb_none)
            : ValueEngineParameter(name, underlying, init, std::move(cb)), m_min(min), m_max(max) {
        }

        [[nodiscard]] std::string
        uci_declare() const override {
            return fmt::format("option name {} type spin default {} min {} max {}", m_name, m_init, m_min, m_max);
        }

        bool
        parse(const std::string& v) override {
            try {
                const int val = std::stoi(v);
                if (val < m_min || val > m_max) {
                    return false;
                }
                m_value = val;
                return EngineParameter::parse(v);
            } catch (...) {
                return false;
            }
        }

        [[nodiscard]] std::string
        value_str() const override {
            return std::to_string(m_value);
        }

      private:
        int m_min, m_max;
    };

    class EngineParamCombo final : public ValueEngineParameter<std::string> {
      public:
        EngineParamCombo(const std::string&       name,
                         std::string&             underlying,
                         std::string              def,
                         std::vector<std::string> choices,
                         UciCallbacT              cb = uci_cb_none)
            : ValueEngineParameter(name, underlying, std::move(def), std::move(cb)), m_choices(std::move(choices)) {
        }

        [[nodiscard]] std::string
        uci_declare() const override {
            std::ostringstream ss;
            fmt::print(ss, "option name {} type combo default {}", m_name, m_init);
            for (const auto& c : m_choices) {
                fmt::print(ss, " var {}", c);
            }

            return ss.str();
        }

        bool
        parse(const std::string& v) override {
            if (std::ranges::find(m_choices, v) != m_choices.end()) {
                m_value = v;
                return EngineParameter::parse(v);
            }
            return false;
        }

        [[nodiscard]] std::string
        value_str() const override {
            return m_value;
        }

      private:
        std::vector<std::string> m_choices;
    };

    class EngineParamString final : public ValueEngineParameter<std::string> {
      public:
        explicit EngineParamString(const std::string& name,
                                   std::string&       underlying,
                                   std::string        init = "",
                                   UciCallbacT        cb   = uci_cb_none)
            : ValueEngineParameter(name, underlying, std::move(init), std::move(cb)) {
        }

        [[nodiscard]] std::string
        uci_declare() const override {
            return fmt::format("option name {} type string default {}", m_name, m_init);
        }

        bool
        parse(const std::string& v) override {
            m_value = v;
            return EngineParameter::parse(v);
        }

        [[nodiscard]] std::string
        value_str() const override {
            return m_value;
        }
    };

    class EngineParamButton final : public EngineParameter {
      public:
        explicit EngineParamButton(const std::string& name, UciCallbacT cb = uci_cb_none)
            : EngineParameter(name, std::move(cb)) {
        }

        [[nodiscard]] std::string
        uci_declare() const override {
            return fmt::format("option name {} type button", m_name);
        }

        bool
        parse(const std::string& v) override {
            return EngineParameter::parse(v);
        }

        [[nodiscard]] std::string
        value_str() const override {
            return "<button>";
        }
    };

    class EngineParameterHandler {
      public:
        template <typename T, typename... Args>
        T*
        add(Args&&... args) {
            auto param                  = std::make_unique<T>(std::forward<Args>(args)...);
            T*   ptr                    = param.get();
            m_params_map[param->name()] = ptr;
            m_params.push_back(std::move(param));
            return ptr;
        }

        void
        print_uci_options(FILE* os) const {
            for (const auto& p : m_params) {
                fmt::print(os, "{}\n", p->uci_declare());
            }
        }

        bool
        set(const std::string& name, const std::string& value) {
            auto it = m_params_map.find(name);
            if (it == m_params_map.end()) {
                return false;
            }
            return it->second->parse(value);
        }

        bool
        handle_setoption(const std::string& command) {
            std::istringstream iss(command);
            std::string        token;
            iss >> token;
            iss >> token;

            if (token != "name") {
                return false;
            }

            std::string name, value;
            std::string word;

            while (iss >> word) {
                if (word == "value") {
                    break;
                }
                if (!name.empty()) {
                    name += " ";
                }
                name += word;
            }

            std::getline(iss, value);
            if (!value.empty() && value[0] == ' ') {
                value.erase(0, 1);
            }

            if (name.empty()) {
                return false;
            }

            auto it = m_params_map.find(name);
            if (it == m_params_map.end()) {
                return false;
            }

            if (value.empty()) {
                value = "true";
            }
            return it->second->parse(value);
        }

      private:
        std::vector<std::unique_ptr<EngineParameter>>     m_params{};
        std::unordered_map<std::string, EngineParameter*> m_params_map{};
    };

    class UCIEngine {

        enum State {
            Waiting,
            Searching,
            Pondering,
            Terminated,
        };

        struct Parameters {
            int                      hash_size{};
            int                      threads{};
            std::string              tb_path{};
            TimeManager::Params      tm{};
            SearchThread::Parameters tunables;
        };

        Parameters             m_params{};
        State                  m_state{Waiting};
        Positions              m_pos{};
        std::jthread           m_worker;
        SearchThreadHandler    m_thread_handler{};
        EngineParameterHandler m_param_handler{};
        TT                     m_tt{}; // lifetime for the whole life of the engine
        nnue::NetworkHandle    m_handle{};

      public:
        explicit UCIEngine(const bool enable_tuning = false) {
            m_param_handler.add<EngineParamSpin>("Hash Size", m_params.hash_size, 64, 64, 512, [this]() {
                m_tt.reset();
                m_tt.init(m_params.hash_size);
                fmt::print(stdout, "info string Hash Size {}\n", m_params.hash_size);
                return true;
            });
            m_param_handler.add<EngineParamSpin>(
                "Threads", m_params.threads, 1, 1, std::thread::hardware_concurrency());
            m_param_handler.add<EngineParamString>("SyzygyPath", m_params.tb_path, "", [this]() {
                const bool val = init_tb(m_params.tb_path);
                if (val) {
                    fmt::print(stdout, "info string set tb path\n");
                }
                return val;
            });
            m_param_handler.add<EngineParamButton>("Clear Hash", [this]() {
                m_tt.reset();
                fmt::print(stdout, "info string Hash cleared\n");
                return true;
            });
            m_param_handler.add<EngineParamButton>("Tune", [this]() {
                m_handle.tune();
                return true;
            });
            if (enable_tuning) // to tune magic values
            {
                m_param_handler.add<EngineParamSpin>(
                    "AspWin min depth", m_params.tunables.aspiration_window_activation_depth, 7, 1, 10);
            }
            m_tt.init(m_params.hash_size);
            if (auto err = m_pos.set_fen<true>(start_fen); !err) {
                throw std::runtime_error("Failed to set start fen: " + err.error());
            }
        }

        void
        uci() const {
            if (m_state != Waiting) return;
            fmt::print(stdout,
                       "id name ChePP\n"
                       "id author pauxav06\n");
            m_param_handler.print_uci_options(stdout);
            fmt::print(stdout, "uciok\n");
        }

        void
        isready() const {
            if (m_state != Waiting) return;
            fmt::print(stdout, "readyok\n");
        }

        void
        ucinewgame() {
            if (m_state != Waiting) {
                return;
            }
            m_tt.reset();
            m_pos.clear();
            m_pos.set_fen(start_fen);
        }

        static tl::expected<std::vector<Move>, std::string>
        parse_moves(std::istringstream& iss, Position& movegen) {
            std::string       token;
            std::vector<Move> moves{};
            while (iss >> token) {
                auto move = Move::from_uci(token,
                                           {.pieces          = movegen.pieces(),
                                            .ep_square       = movegen.ep_square(),
                                            .castling_rights = movegen.castling_rights()});
                if (!move || !movegen.is_valid(*move)) {
                    return tl::unexpected(fmt::format("invalid move {}", token));
                }
                moves.push_back(*move);
                movegen.do_move(*move);
            }
            return moves;
        }

        void
        position(const std::string& cmd) {
            if (m_state != Waiting) {
                return;
            }

            std::istringstream iss(cmd);
            std::string        token;
            iss >> token;
            assert(token == "position");

            std::string type;
            iss >> type;
            std::vector<Move> moves;
            std::string       fen;

            if (type == "startpos") {
                fen = std::string(start_fen);
            } else if (type == "fen") {
                std::string part;
                for (int i = 0; i < 6 && iss >> part; ++i) {
                    if (i) fen += " ";
                    fen += part;
                }
            }
            auto movegen{Fen::from_string(fen).transform(&Position::from_fen)};

            if (!movegen) {
                fmt::print(stderr, "Invalid fen {}\n", movegen.error());
                return;
            }
            if (iss >> token && token == "moves") {
                auto err = parse_moves(iss, *movegen);
                if (!err) {
                    fmt::print(stderr, "Invalid move {}\n", err.error());
                    return;
                }
                moves = *err;
            }

            Positions temp{};
            if (auto err = temp.set_fen<true>(fen, moves); !err) {
                fmt::print(stderr, "Invalid fen: {}\n", err.error());
                return;
            }
            m_pos = temp;
        }

        void
        go(const std::string& cmd) {
            if (m_state != Waiting) return;

            TimeManager::UCIConstraints constraints;
            std::istringstream          iss(cmd);
            std::string                 token;

            while (iss >> token) {
                if (token == "wtime")
                    iss >> constraints.time[WHITE];
                else if (token == "btime")
                    iss >> constraints.time[BLACK];
                else if (token == "winc")
                    iss >> constraints.inc[WHITE];
                else if (token == "binc")
                    iss >> constraints.inc[BLACK];
                else if (token == "movestogo")
                    iss >> constraints.moves_to_go;
                else if (token == "depth") {
                    iss >> constraints.depth;
                } else if (token == "movetime") {
                    iss >> constraints.move_time;
                }
            }

            m_thread_handler.set(
                m_params.threads, m_params.tunables, m_params.tm, constraints, &m_tt, m_pos, m_handle.get());
            m_worker = std::jthread([&]() {
                m_thread_handler.start();
                m_state = Waiting;
            });

            m_state = Searching;
        }

        void
        eval() const {
            using namespace chepp::nnue;
            fmt::print(stdout, "{}\n", m_pos.last().to_string());
            auto network = m_handle.get();
            network->init(m_pos.last());
            fmt::print(stdout, "Evaluation for {} (cp):\n", m_pos.last().side_to_move());
            network->dbg_uci(m_pos.last().side_to_move());
        }

        void
        nnue_cfg() {
            // stdout << "info string " << chepp::nnue::Arch::print_cfg();
        }

        static void
        bench() {
            /**
            static constexpr int n_positions = 10; // just to not get messed up by cache
            Positions            pos;
            auto                 err = pos.set_fen(start_fen);
            chepp::nnue::Accumulators   accum{pos.last()};
            for (int i = 0; i < n_positions; i++) {
                auto moves = gen_legal(pos.last());
                pos.do_move(moves[0]);
                accum.do_move(pos[pos.ply() - 1], pos.last());
            }
            auto& rng     = prng::thread_local_gen();
            auto  distrib = std::uniform_int_distribution<int8_t>(0, 9);

            volatile int64_t tot{0};
            constexpr size_t n_iterations = 1'000'000;

            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < n_iterations; i++) {
                tot += chepp::nnue::network.evaluate(accum[distrib(rng)], WHITE);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

            stdout << "Time for " << n_iterations << " evaluations: " << ms.count()
                      << " (" << ns.count() / n_iterations << " ns/it) (tot "<<  tot <<  ")"
                      << std::endl;

            start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < n_iterations; i++) {
                chepp::nnue::Accumulator::bench_refresh(pos[distrib(rng)]);
            }
            end = std::chrono::high_resolution_clock::now();
            ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

            stdout << "time for " << n_iterations << " refreshes: " << ms.count()
                      << " (" << ns.count() / n_iterations << " ns/it) (tot "<<  tot <<  ")"
                      << std::endl;

                      **/
        }

        void
        stop() {
            m_thread_handler.stop_all();
            if (m_worker.joinable()) m_worker.join();
        }
        void
        handle_command(const std::string& line) {
            if (line == "uci") {
                uci();
            } else if (line == "isready") {
                isready();
            } else if (line == "ucinewgame") {
                ucinewgame();
            } else if (line.rfind("position", 0) == 0) {
                position(line);
            } else if (line.rfind("go", 0) == 0) {
                go(line);
            } else if (line.rfind("setoption", 0) == 0) {
                if (!m_param_handler.handle_setoption(line))
                    fmt::print(stderr, "info string Unknown option or invalid value\n");
            } else if (line == "evaluate" || line == "eval") {
                eval();
            } else if (line == "bench") {
                bench();
            } else if (line == "print") {
                fmt::print(stdout, "{}\n", m_pos.last().to_string());
            } else if (line == "cfg") {
                nnue_cfg();
            } else if (line == "stop") {
                stop();
            } else if (line == "quit") {
                stop();
                m_state = Terminated;
            } else if (line == "perft") {
                perft_divide(m_pos.last(), 7);
            }
        }

        int
        loop(std::istream& in) {
            std::string line;
            while (m_state != Terminated && std::getline(in, line)) {
                handle_command(line);
                std::fflush(stdout);
            }

            return 0;
        }
    };
} // namespace chepp

#endif // CHEPP_UCI_H
