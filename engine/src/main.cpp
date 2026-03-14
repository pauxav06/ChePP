#include "engine.h"

#include <iostream>
using namespace chepp;

int main(int argc, char** argv)
{
    UCIEngine          engine{false};
    if (argc == 1) {
        return engine.loop(std::cin);
    } else if (argc >= 2) {
        std::stringstream oss{};
        for (int i{1}; i < argc; ++i) {
            oss << argv[i] << std::endl;
        }
        return engine.loop(oss);
    }
    return 0;
}
