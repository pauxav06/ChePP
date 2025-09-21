//
// Created by paul on 9/6/25.
//

#ifndef CHEPP_UCI_H
#define CHEPP_UCI_H

#include "ChePP/engine/position.h"
#include "ChePP/engine/search.h"
#include "ChePP/engine/tm.h"
#include "tb.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using UciCallbacT = std::function<bool()>;
inline auto uci_cb_none = [] () { return true; };

class EngineParameter {
public:
    explicit EngineParameter(std::string name, UciCallbacT cb = uci_cb_none)
        : m_name(std::move(name)) , m_cb(std::move(cb)) {}
    virtual ~EngineParameter() = default;

    [[nodiscard]] const std::string& name() { return m_name; };
    [[nodiscard]] virtual std::string uci_declare() const = 0;
    virtual bool parse([[maybe_unused]] const std::string& value)
    {
        return m_cb();
    };
    [[nodiscard]] virtual std::string value_str() const = 0;
protected:
    std::string m_name;
    UciCallbacT m_cb;
};

template <typename T>
class ValueEngineParameter : public EngineParameter {
public:
    explicit ValueEngineParameter(std::string name, T& underlying, T  init, UciCallbacT cb = uci_cb_none)
    : EngineParameter(std::move(name), std::move(cb)), m_init(std::move(init)), m_value(underlying)
    {
        m_value = m_init;
    }

    ~ValueEngineParameter() override = default;

    [[nodiscard]] T value() const { return m_value; }

protected:
    T m_init{};
    T& m_value{};
};

class EngineParamCheck final : public ValueEngineParameter<bool> {
public:
    explicit EngineParamCheck(std::string name, bool& underlying, const bool def = false, UciCallbacT cb = uci_cb_none)
        : ValueEngineParameter(std::move(name), underlying, def, std::move(cb)) {}

    [[nodiscard]] std::string uci_declare() const override {
        return "option name " + m_name + " type check default " + (m_init ? "true" : "false");
    }

    bool parse(const std::string& v) override {
        if (v == "true" || v == "1")
        {
            m_value = true;
            return EngineParameter::parse(v);
        }
        if (v == "false" || v == "0")
        {
            m_value = false;
            return EngineParameter::parse(v);
        }
        return false;
    }

    [[nodiscard]] std::string value_str() const override { return m_value ? "true" : "false"; }
};

class EngineParamSpin final : public ValueEngineParameter<int> {
public:
    EngineParamSpin(std::string name, int& underlying, const int init, const int min, const int max, UciCallbacT cb = uci_cb_none)
        : ValueEngineParameter(std::move(name), underlying, init, std::move(cb)), m_min(min), m_max(max) {}

    [[nodiscard]] std::string uci_declare() const override {
        return "option name " + m_name + " type spin default " +
               std::to_string(m_init) + " min " + std::to_string(m_min) +
               " max " + std::to_string(m_max);
    }

    bool parse(const std::string& v) override {
        try {
            const int val = std::stoi(v);
            if (val < m_min || val > m_max) return false;
            m_value = val;
            return EngineParameter::parse(v);
        } catch (...) { return false; }
    }

    [[nodiscard]] std::string value_str() const override { return std::to_string(m_value); }

private:
    int m_min, m_max;
};

class EngineParamCombo final : public ValueEngineParameter<std::string> {
public:
    EngineParamCombo(std::string name, std::string& underlying, std::string def, std::vector<std::string> choices, UciCallbacT cb = uci_cb_none)
        : ValueEngineParameter(std::move(name), underlying, std::move(def), std::move(cb)), m_choices(std::move(choices)) {}

    [[nodiscard]] std::string uci_declare() const override {
        std::ostringstream ss;
        ss << "option name " << m_name << " type combo default " << m_init;
        for (const auto& c : m_choices) ss << " var " << c;
        return ss.str();
    }

    bool parse(const std::string& v) override {
        if (std::ranges::find(m_choices, v) != m_choices.end()) {
            m_value = v;
            return EngineParameter::parse(v);
        }
        return false;
    }

    [[nodiscard]] std::string value_str() const override { return m_value; }

private:
    std::vector<std::string> m_choices;
};

class EngineParamString final : public ValueEngineParameter<std::string> {
public:
    explicit EngineParamString(std::string name, std::string& underlying, std::string init = "", UciCallbacT cb = uci_cb_none)
        : ValueEngineParameter(std::move(name), underlying, std::move(init), std::move(cb)) {}

    [[nodiscard]] std::string uci_declare() const override {
        return "option name " + m_name + " type string default " + m_init;
    }

    bool parse(const std::string& v) override
    {
        m_value = v;
        return EngineParameter::parse(v);
    }

    [[nodiscard]] std::string value_str() const override { return m_value; }
};


class EngineParamButton final : public EngineParameter {
public:

    explicit EngineParamButton(const std::string& name, UciCallbacT cb = uci_cb_none)
        : EngineParameter(name, std::move(cb)) {}

    [[nodiscard]] std::string uci_declare() const override {
        return "option name " + m_name + " type button";
    }

    bool parse(const std::string& v) override {
        return EngineParameter::parse(v);
    }

    [[nodiscard]] std::string value_str() const override {
        return "<button>";
    }
};

class EngineParameterHandler {
public:
    template <typename T, typename... Args>
    T* add(Args&&... args) {
        auto param = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = param.get();
        m_params_map[param->name()] = ptr;
        m_params.push_back(std::move(param));
        return ptr;
    }

    void print_uci_options(std::ostream& os) const {
        for (const auto& p : m_params)
           os << p->uci_declare() << "\n";
    }

    bool set(const std::string& name, const std::string& value) {
        auto it = m_params_map.find(name);
        if (it == m_params_map.end()) return false;
        return it->second->parse(value);
    }

    bool handle_setoption(const std::string& command) {
        std::istringstream iss(command);
        std::string token;
        iss >> token;
        iss >> token;


        if (token != "name") return false;

        std::string name, value;
        std::string word;


        while (iss >> word) {
            if (word == "value") break;
            if (!name.empty()) name += " ";
            name += word;
        }

        std::getline(iss, value);
        if (!value.empty() && value[0] == ' ') value.erase(0,1);

        if (name.empty()) return false;

        auto it = m_params_map.find(name);
        if (it == m_params_map.end()) return false;

        if (value.empty()) value = "true";
        return it->second->parse(value);
    }

private:
    std::vector<std::unique_ptr<EngineParameter>> m_params{};
    std::unordered_map<std::string, EngineParameter*> m_params_map{};
};




class UCIEngine {

    enum State
    {
        Waiting,
        Searching,
        Pondering
    };

    struct Parameters
    {
        int hash_size{};
        int threads{};
        std::string tb_path{};
        TimeManager::Params tm{};
        SearchThread::Parameters tunables;
        EngineParameterHandler handler{};
    };


    Parameters m_params{};
    State m_state{Waiting};
    Positions m_pos{};
    std::jthread m_worker;
    SearchThreadHandler m_handler{};
    TT m_tt{}; // lifetime for the whole life of the engine

public:
    explicit UCIEngine(const bool enable_tuning = false) {
        m_params.handler.add<EngineParamSpin>("Hash Size", m_params.hash_size, 64, 64, 512, [this] ()
        {
            m_tt.reset();
            m_tt.init(m_params.hash_size);
            std::cout << std::format("info string Hash Resized to {}", m_params.hash_size) << std::endl;
            return true;
        });
        m_params.handler.add<EngineParamSpin>("Threads", m_params.threads, 1, 1, std::thread::hardware_concurrency());
        m_params.handler.add<EngineParamString>("SyzygyPath", m_params.tb_path, "", [this] ()
        {
            const bool val = init_tb(m_params.tb_path);
            if (val) std::cout << "info string set tb path" << std::endl;
            return val;

        });
        m_params.handler.add<EngineParamButton>("Clear Hash", [this]() {
            m_tt.reset();
            std::cout << "info string Hash cleared" << std::endl;
            return true;
        });
        if (enable_tuning)
        {
            m_params.handler.add<EngineParamSpin>("AspWin min depth", m_params.tunables.aspiration_window_activation_depth, 7, 1, 10);
        }
        m_tt.init(m_params.hash_size);
        m_pos.set_fen(start_fen);
    }

    void uci() const
    {
        if (m_state != Waiting) return;
        std::cout << "id name ChePP\n";
        std::cout << "id author Paul\n";
        m_params.handler.print_uci_options(std::cout);
        std::cout << "uciok" << std::endl;
    }

    void isready() const {
        if (m_state != Waiting) return;
        std::cout << "readyok" << std::endl ;
    }

    void ucinewgame()
    {
        if (m_state != Waiting) return;
        m_tt.reset();
        m_pos.clear();
        m_pos.set_fen(start_fen);
    }

    MoveList parse_moves(std::istringstream& iss, Position& movegen) {
        std::string token;
        MoveList moves{};
        while (iss >> token) {
            auto move = Move::from_uci(token, {
                .pieces = movegen.pieces(),
                .ep_square = movegen.ep_square(),
                .castling_rights = movegen.castling_rights()
            });
            if (!move || !movegen.is_valid_move(*move)) {
                throw std::invalid_argument(std::format("invalid move {}", token));
            }
            moves.push_back(*move);
            movegen.do_move(*move);
        }
        return moves;
    }

    void position(const std::string& cmd) {
        if (m_state != Waiting) return;

        std::istringstream iss(cmd);
        std::string token;
        iss >> token;
        assert(token == "position");

        m_pos.clear();

        std::string type;
        iss >> type;
        MoveList moves;
        std::string fen;

        try {
            Position movegen;
            if (type == "startpos")
            {
                fen = start_fen;
            }
            else if (type == "fen")
            {
                std::string part;
                for (int i = 0; i < 6 && iss >> part; ++i) {
                    if (i) fen += " ";
                    fen += part;
                }
            }
            if (!movegen.from_fen(fen)) {
                throw std::invalid_argument("invalid fen");
            }
            if (iss >> token && token == "moves") {
                moves = parse_moves(iss, movegen);
            }
            m_pos.set_fen(fen, moves);
        } catch (const std::exception& e) {
            std::cerr << "position command error: " << e.what() << '\n';
        }
    }

    void go(const std::string& cmd)
    {
        if (m_state != Waiting) return;
        TimeManager::UCIConstraints constraints;
        std::istringstream iss(cmd);
        std::string token;

        while (iss >> token) {
            if (token == "wtime") iss >> constraints.time[WHITE];
            else if (token == "btime") iss >> constraints.time[BLACK];
            else if (token == "winc")  iss >> constraints.inc[WHITE];
            else if (token == "binc")  iss >> constraints.inc[BLACK];
            else if (token == "movestogo") iss >> constraints.moves_to_go;
            else if (token == "depth") { iss >> constraints.depth;}
            else if (token == "movetime") { iss >> constraints.move_time; }

        }

        m_handler.set(m_params.threads, m_params.tunables, m_params.tm, constraints, &m_tt, m_pos);
        m_worker = std::jthread([&]()
        {
            m_handler.start();
            m_state = Waiting;
        });

        m_state = Searching;
    }

    void eval() const
    {
        std::cout << m_pos.last() << std::endl;
        const Accumulator accum{m_pos.last()};
        std::cout << "Evaluation for " << m_pos.last().side_to_move() << " (cp): " << std::endl;
        accum.evaluate_uci(m_pos.last().side_to_move());
    }


    void stop()
    {
        m_handler.stop_all();
        if (m_worker.joinable()) m_worker.join();
    }

    int loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
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
                if (!m_params.handler.handle_setoption(line))
                    std::cerr << "info string Unknown option or invalid value\n" << std::endl;
            } else if (line == "evaluate" || line == "eval") {
                eval();
            } else if (line  == "print") {
                std::cout << m_pos.last() << std::endl;
            } else if (line == "stop") {
                stop();
            } else if (line == "quit") {
                stop();
                break;
            }
        }
        return 0;
    }
};



#endif // CHEPP_UCI_H
