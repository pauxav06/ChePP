#include "ChePP/engine/movegen.h"
#include "ChePP/engine/position.h"

#include <array>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <random>
#include <cmath>
#include "ChePP/engine/pgn.h"

struct EngineProcess {
    std::string path;
    pid_t pid;
    std::unique_ptr<FILE, decltype(&fclose)> in;
    std::unique_ptr<FILE, decltype(&fclose)> out;

    EngineProcess(std::string p, const pid_t child, FILE* i, FILE* o)
        : path(std::move(p)), pid(child), in(i, fclose), out(o, fclose) {}

    ~EngineProcess() {
        if (pid > 0) {
            int status = 0;
            kill(pid, SIGTERM);
            waitpid(pid, &status, 0);
        }
    }
};

EngineProcess start_engine(const std::string& path) {
    int in_pipe[2];
    int out_pipe[2];
    if (pipe(in_pipe) == -1 || pipe(out_pipe) == -1)
        throw std::runtime_error("pipe() failed");

    pid_t pid = fork();
    if (pid == -1)
        throw std::runtime_error("fork() failed");

    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[1]);
        close(out_pipe[0]);
        execl(path.c_str(), path.c_str(), static_cast<char*>(nullptr));
        perror("execl failed");
        _exit(1);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);

    FILE* child_stdin  = fdopen(in_pipe[1], "w");
    FILE* child_stdout = fdopen(out_pipe[0], "r");
    if (!child_stdin || !child_stdout)
        throw std::runtime_error("fdopen() failed");

    return {path, pid, child_stdin, child_stdout};
}

struct UCIEngine {
    EngineProcess proc;

    explicit UCIEngine(const std::string& path) : proc(start_engine(path)) {
        send("uci");
        wait_for("uciok");
    }

    void send(const std::string& cmd) const {
        fprintf(proc.in.get(), "%s\n", cmd.c_str());
        fflush(proc.in.get());
    }

    std::string wait_for(const std::string& keyword, int timeout_ms = 5000) const {
        const int fd = fileno(proc.out.get());
        pollfd pfd{fd, POLLIN, 0};

        std::string buffer;
        auto start = std::chrono::steady_clock::now();
        char tmp[256];

        while (true) {
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
            if (elapsed > timeout_ms)
                throw std::runtime_error("Timeout waiting for: " + keyword);

            int ret = poll(&pfd, 1, timeout_ms - elapsed);
            if (ret <= 0) continue;

            ssize_t n = read(fd, tmp, sizeof(tmp)-1);
            if (n > 0) {
                tmp[n] = '\0';
                buffer += tmp;

                size_t pos;
                while ((pos = buffer.find('\n')) != std::string::npos) {
                    std::string line = buffer.substr(0, pos);
                    buffer.erase(0, pos + 1);
                    if (line.find(keyword) != std::string::npos)
                        return line;
                }
            } else if (n == 0) {
                throw std::runtime_error("Engine pipe closed");
            }
        }
    }


    void is_ready() const {
        send("isready");
        wait_for("readyok");
    }

    void new_game() const { send("ucinewgame"); }

    void set_position(const std::string& fen, const std::string& moves = "") const {
        std::string cmd = "position fen " + fen;
        if (!moves.empty()) cmd += " moves " + moves;
        send(cmd);
    }

    [[nodiscard]] std::string bestmove(int depth = 10) const {
        send("go depth " + std::to_string(depth));
        const auto line = wait_for("bestmove", 100000);
        std::istringstream iss(line);
        std::string token, move;
        iss >> token >> move;
        return move;
    }
};

struct GameManager {
    std::vector<Position> positions;
    std::string moves_uci;

    explicit GameManager(const std::string& fen) {
        positions.reserve(MAX_PLY);
        positions.emplace_back();
        positions.back().from_fen(fen);
    }

    bool is_finished(Result& result) {
        if (Position::is_repetition(positions)) {
            result = DRAW;
            return true;
        }
        if (gen_legal(positions.back()).empty()) {
            Color side = positions.back().side_to_move();
            result = positions.back().checkers(side) ? Result{~side} : DRAW;
            return true;
        }
        return false;
    }

    bool apply_move(const std::string& uci_move) {
        const auto move = move_from_uci(uci_move, positions.back());
        if (!move) return false;
        positions.emplace_back(positions.back());
        positions.back().do_move(*move);
        moves_uci += uci_move + " ";
        return true;
    }

    [[nodiscard]] const std::string& moves() const { return moves_uci; }
};

uint64_t perft_count(Position& pos, int depth) {
    if (depth == 0) return 1;
    uint64_t nodes = 0;
    for (const auto moves = gen_legal(pos); auto [m,s] : moves) {
        Position next = pos;
        next.do_move(m);
        nodes += perft_count(next, depth - 1);
    }
    return nodes;
}

void perft_sequences(const Position& pos, const int depth,
                     std::vector<std::string>& current,
                     std::vector<std::vector<std::string>>& sequences) {
    if (depth == 0) {
        sequences.push_back(current);
        return;
    }
    auto moves = gen_legal(pos);
    for (auto& [m,s] : moves) {
        Position next = pos;
        next.do_move(m);
        current.push_back(m.to_string());
        perft_sequences(next, depth - 1, current, sequences);
        current.pop_back();
    }
}

int find_depth_for_tests(const std::string& fen, int N) {
    Position pos;
    pos.from_fen(fen);
    for (int d = 1; d < 10; d++) {
        if (perft_count(pos, d) >= static_cast<uint64_t>(N)) return d;
    }
    throw std::runtime_error("Not enough positions up to depth 10");
}

std::vector<std::vector<std::string>> assign_openings(const std::string& fen, int N) {
    int depth = find_depth_for_tests(fen, N);
    Position pos;
    pos.from_fen(fen);
    std::vector<std::vector<std::string>> sequences;
    std::vector<std::string> current;
    perft_sequences(pos, depth, current, sequences);
    if (static_cast<int>(sequences.size()) < N)
        throw std::runtime_error("Not enough sequences generated");
    sequences.resize(N);
    return sequences;
}

struct BenchmarkResult {
    Result result;
    std::string pgn;
};

BenchmarkResult benchmark_with_opening(const std::string& fen,
                                       const std::vector<std::string>& opening,
                                       const std::string& eng1_path,
                                       const std::string& eng2_path, const int depth = 10)
{
    UCIEngine white(eng1_path);
    UCIEngine black(eng2_path);

    white.is_ready();
    black.is_ready();
    white.new_game();
    black.new_game();

    GameManager game(fen);

    for (auto& mv : opening) {
        if (!game.apply_move(mv))
            throw std::runtime_error("Invalid opening move: " + mv);
    }
    white.set_position(fen, game.moves());
    black.set_position(fen, game.moves());

    Result result;
    while (!game.is_finished(result)) {
        auto& engine = game.positions.back().side_to_move() == WHITE ? white : black;
        engine.set_position(fen, game.moves());
        if (auto move = engine.bestmove(depth); !game.apply_move(move))
            throw std::runtime_error("Illegal move from engine: " + move);
    }

    using Fields = PGN::Fields<PGN::Event, PGN::Site, PGN::Date, PGN::Round, PGN::White, PGN::Black, PGN::Result>;
    return BenchmarkResult{result,
        PGN::to_pgn(game.positions,
        Fields{"Benchmark", "ThinkPad", Date{2025, 10, 7}.to_string(), 10, "WhiteEngine", "BlackEngine", result.to_string().cbegin()})
    };
}


int main(int argc, char** argv) {
    constexpr int N = 8;
    constexpr int n_threads = 8;

    auto openings = assign_openings(start_fen, N);


    BenchmarkResult results[2*N];

    std::ofstream pgn_file("benchmark_results.pgn");
    if (!pgn_file) throw std::runtime_error("Failed to open benchmark_results.pgn");

    std::mutex pgn_mutex;

    int chunk_size = (N + n_threads - 1) / n_threads;
    std::vector<std::jthread> threads;

    for (int t = 0; t < n_threads; t++) {
        int start_idx = t * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, N);

        threads.emplace_back([start_idx, end_idx, &pgn_file, &results, &openings, &pgn_mutex]() {
            for (int i = start_idx; i < end_idx; i++) {
                results[2*i] = benchmark_with_opening(
                    start_fen, openings[i],
                    "./versions/ChePP_3474e90", "./versions/ChePP_3474e90", 8
                );

                results[2*i+1] = benchmark_with_opening(
                    start_fen, openings[i],
                    "./versions/ChePP_3474e90", "./versions/ChePP_3474e90", 8
                );

                std::lock_guard lock(pgn_mutex);
                pgn_file << results[2*i].pgn << "\n\n";
                pgn_file << results[2*i+1].pgn << "\n\n";
            }
        });
    }

    for (auto& t : threads) t.join();

    EnumArray<Result, int> wins{};
    for (auto& r : results) {
        wins[r.result]++;
    }

    int decisive_games = wins[WIN_WHITE] + wins[WIN_BLACK];
    double win_rate = decisive_games > 0 ? static_cast<double>(wins[WIN_WHITE]) / decisive_games : 0.5;

    double z = 1.96;
    double se = (decisive_games > 0) ? std::sqrt(win_rate * (1 - win_rate) / decisive_games) : 0.0;
    double ci_low  = std::max(0.0, win_rate - z * se);
    double ci_high = std::min(1.0, win_rate + z * se);

    bool significant = (ci_low > 0.5 || ci_high < 0.5);

    std::cout << "============================================\n";
    std::cout << "Benchmark complete: " << 2*N << " games\n";
    std::cout << "White wins: " << wins[WIN_WHITE] << "\n";
    std::cout << "Black wins: " << wins[WIN_BLACK] << "\n";
    std::cout << "Draws     : " << wins[DRAW] << "\n";
    std::cout << "Win rate (white in decisive games): "
              << (100.0 * win_rate) << "%\n";
    std::cout << "95% CI: [" << 100.0*ci_low << "%, " << 100.0*ci_high << "%]\n";
    std::cout << "Significant vs 50%: " << (significant ? "YES" : "NO") << "\n";
    std::cout << "PGNs stored in benchmark_results.pgn\n";
    std::cout << "============================================\n";
}