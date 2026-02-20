#include "UCI.h"

#include <vector>

int
main() {
    using namespace chepp;
    auto network = nnue::Arch::make_network();
    auto pos     = Position{*Fen::from_string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1")};
    network.init(pos);
    std::cout << network.forward(WHITE) << std::endl;

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
