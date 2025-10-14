#include "UCI.h"

int main() {
    UCIEngine          engine{false};
    std::istringstream bench_input("ucinewgame\n"
                                   "position startpos\n"
                                   "go depth 20\n"
                                   "quit\n");
    // engine.loop(bench_input);
    // engine = UCIEngine();
    engine.loop(std::cin);
    return 0;
}
