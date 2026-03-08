#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "bitboard.h"
#include "format.h"
#include <argparse/argparse.hpp>

using namespace chepp;
namespace fs = std::filesystem;

inline bool
write_magic_bishops(FILE* out) {
    const auto b = std::make_unique<movegen::detail::Magics<BISHOP>>();
    b->init();
    std::fwrite(b.get(), 1, sizeof(*b), out);
    return true;
}

inline bool
write_magic_rooks(FILE* out) {
    auto b = std::make_unique<movegen::detail::Magics<ROOK>>();
    b->init();
    std::fwrite(b.get(), 1, sizeof(*b), out);
    return true;
}

inline bool
write_lines(FILE* out) {
    auto l = std::make_unique<movegen::detail::lines_type>(std::in_place, movegen::detail::compute_lines);
    std::fwrite(l.get(), 1, sizeof(*l), out);
    return true;
}

inline bool
write_from_to(FILE* out) {
    auto l = std::make_unique<movegen::detail::lines_type>(std::in_place, movegen::detail::compute_from_to);
    std::fwrite(l.get(), 1, sizeof(*l), out);
    return true;
}

int
main(int argc, char** argv) {
    std::unordered_map<std::string, std::function<bool(FILE*)>> targets;
    targets.emplace("magic_rooks", write_magic_rooks);
    targets.emplace("magic_bishops", write_magic_bishops);
    targets.emplace("lines", write_lines);
    targets.emplace("from_to", write_from_to);

    argparse::ArgumentParser program("dump");
    program.add_argument("--output").required().help("Path to output file");
    auto targets_arg = program.add_argument("--target").required();
    for (const auto& [name, _] : targets) {
        targets_arg.add_choice(name);
    }

    program.parse_args(argc, argv);

    fs::path path   = program.get<std::string>("--output");
    auto     target = program.get<std::string>("--target");

    try {
        file_ptr out(fopen(path.string().c_str(), "wb"));
        if (!out) {
            throw std::runtime_error(fmt::format("failed to open file {}", path.c_str()));
        }
        targets.at(target)(out.get());
    } catch (const std::exception& e) {
        fmt::println(stderr, "{}", e.what());
        fs::remove(path);
        return 1;
    }

    return 0;
}