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

inline auto uci_cb_none = [] () { return true; };

class EngineParameter {
public:
    explicit EngineParameter(std::string name, std::function<bool()> cb = uci_cb_none)
        : m_name(std::move(name)) , m_cb(std::move(cb)) {}
    virtual ~EngineParameter() = default;

    [[nodiscard]] const std::string& name() { return m_name; };
    [[nodiscard]] virtual std::string uci_declare() const = 0;
    virtual bool parse(const std::string& value)
    {
        return m_cb();
    };
    [[nodiscard]] virtual std::string value_str() const = 0;
protected:
    std::string m_name;
    std::function<bool()> m_cb;
};

template <typename T>
class ValueEngineParameter : public EngineParameter {
public:
    explicit ValueEngineParameter(std::string name, T& underlying, T  init, std::function<bool()> cb = uci_cb_none)
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
    explicit EngineParamCheck(std::string name, bool& underlying, const bool def = false, std::function<bool()> cb = uci_cb_none)
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
    EngineParamSpin(std::string name, int& underlying, const int init, const int min, const int max, std::function<bool()> cb = uci_cb_none)
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
    EngineParamCombo(std::string name, std::string& underlying, std::string def, std::vector<std::string> choices, std::function<bool()> cb = uci_cb_none)
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
    explicit EngineParamString(std::string name, std::string& underlying, std::string init = "", std::function<bool()> cb = uci_cb_none)
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

    explicit EngineParamButton(const std::string& name, std::function<bool()> cb = uci_cb_none)
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

class EngineParameters {
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
        EngineParameters handler{};
    };

    struct Pos
    {
        Position init_pos{};
        Position last_pos{};
        std::vector<Move> moves{};
    };


    Parameters m_params{};
    State m_state{Waiting};
    Pos m_pos{};
    std::jthread m_worker;
    SearchThreadHandler m_handler{};


public:
    UCIEngine() {
        m_params.handler.add<EngineParamSpin>("Hash Size", m_params.hash_size, 64, 64, 512);
        m_params.handler.add<EngineParamSpin>("Threads", m_params.threads, 1, 1, std::thread::hardware_concurrency());
        m_params.handler.add<EngineParamString>("Tb Path", m_params.tb_path, "", [this] ()
        {
            bool val = init_tb(m_params.tb_path);
            if (val) std::cout << "info string set tb path" << std::endl;
            return val;

        });

        m_params.handler.add<EngineParamButton>("Clear Hash", []() {
            g_tt.reset();
            std::cout << "info string Hash cleared" << std::endl;
            return true;
        });
        m_pos.init_pos.from_fen(start_fen);
        m_pos.last_pos.from_fen(start_fen);
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

    void ucinewgame() {
        if (m_state != Waiting) return;
        g_tt.reset();
    }

    void position(const std::string& cmd) {
        if (m_state != Waiting) return;
        std::istringstream iss(cmd);
        std::string token;
        iss >> token;

        m_pos.moves.clear();
        m_pos.moves.reserve(MAX_PLY);

        std::string type;
        iss >> type;
        if (type == "startpos") {
            m_pos.init_pos.from_fen(start_fen);
            m_pos.last_pos = m_pos.init_pos;
            if (iss >> token && token == "moves") {
                while (iss >> token) {
                    auto m = Move::from_uci(token, {
                        m_pos.last_pos.pieces(),
                        m_pos.last_pos.ep_square(),
                        m_pos.last_pos.castling_rights()
                    });
                    if (m)
                    {
                        m_pos.moves.push_back(m.value());
                        m_pos.last_pos.do_move(m.value());
                    }
                }
            }
        } else if (type == "fen") {
            std::string fen, part;
            for (int i = 0; i < 6 && iss >> part; ++i) {
                if (i) fen += " ";
                fen += part;
            }
            bool success = m_pos.init_pos.from_fen(fen);
            m_pos.last_pos = m_pos.init_pos;
            if (success)
            {
                std::string is_moves;
                if (iss >> token && token == "moves") {
                    while (iss >> token) {
                        auto m = Move::from_uci(token, {
                            m_pos.last_pos.pieces(),
                            m_pos.last_pos.ep_square(),
                            m_pos.last_pos.castling_rights()
                        });
                        if (m)
                        {
                            m_pos.moves.push_back(m.value());
                            m_pos.last_pos.do_move(m.value());
                        }
                    }
                }
            }
        }
        std::cout << m_pos.init_pos << std::endl;
    }


    void go(const std::string& cmd)
    {
        if (m_state != Waiting) return;
        TimeManager::Constraints constraints;
        std::istringstream iss(cmd);
        std::string token;

        while (iss >> token) {
            if (token == "wtime") iss >> constraints.time[WHITE];
            else if (token == "btime") iss >> constraints.time[BLACK];
            else if (token == "winc")  iss >> constraints.inc[WHITE];
            else if (token == "binc")  iss >> constraints.inc[BLACK];
            else if (token == "movestogo") iss >> constraints.moves_to_go;
            else if (token == "depth") { iss >> constraints.depth; ; }
            else if (token == "movetime") { iss >> constraints.move_time; }

        }

        TimeManager::Params tm_params{};
        TimeManager::InitInfo init_info{};
        init_info.moves_played = m_pos.last_pos.full_move_clock();
        init_info.side = m_pos.last_pos.side_to_move();

        TimeManager tm{ tm_params, init_info, constraints };

        m_handler.set(m_params.threads, tm, m_pos.init_pos, m_pos.moves);
        m_worker = std::jthread([&]()
        {
            m_handler.start();
            m_state = Waiting;
        });

        m_state = Searching;
    }

    void eval() const
    {
        const Accumulator accum{m_pos.last_pos};
        std::cout << "Evaluation for " << m_pos.last_pos.side_to_move() << " (cp): ";
        accum.evaluate_uci(m_pos.last_pos.side_to_move());
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
