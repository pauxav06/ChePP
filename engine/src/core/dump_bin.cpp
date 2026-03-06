#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "bitboard.h"
#include <argparse/argparse.hpp>

using namespace chepp;
namespace fs = std::filesystem;

inline bool
write_magic_bishops(std::ostream& out) {
    const auto b = std::make_unique<movegen::detail::Magics<BISHOP>>();
    b->init();
    std::vector<char> data(sizeof(*b));
    std::memcpy(std::data(data), b.get(), sizeof(*b));
    out.write(std::data(data), static_cast<std::streamsize>(std::size(data)));
    return true;
}

inline bool
write_magic_rooks(std::ostream& out) {
    auto b = std::make_unique<movegen::detail::Magics<ROOK>>();
    b->init();
    std::vector<char> data(sizeof(*b));
    std::memcpy(std::data(data), b.get(), sizeof(*b));
    out.write(std::data(data), static_cast<std::streamsize>(std::size(data)));
    return true;
}

inline bool
write_lines(std::ostream& out) {
    auto              l = std::make_unique<movegen::detail::lines_type>(std::in_place, movegen::detail::compute_lines);
    std::vector<char> data(sizeof(*l));
    std::memcpy(std::data(data), l.get(), sizeof(*l));
    out.write(std::data(data), static_cast<std::streamsize>(std::size(data)));
    return true;
}

inline bool
write_from_to(std::ostream& out) {
    auto l = std::make_unique<movegen::detail::lines_type>(std::in_place, movegen::detail::compute_from_to);
    std::vector<char> data(sizeof(*l));
    std::memcpy(std::data(data), l.get(), sizeof(*l));
    out.write(std::data(data), static_cast<std::streamsize>(std::size(data)));
    return true;
}

int
main(int argc, char** argv) {
    std::unordered_map<std::string, std::function<bool(std::ostream&)>> targets;
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
        std::ofstream out(path, std::ios::binary);
        targets.at(target)(out);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        fs::remove(path);
        return 1;
    }

    return 0;
}