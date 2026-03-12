#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "bitboard.h"
#include "format.h"
#include "span.h"
#include <argparse/argparse.hpp>

using namespace chepp;
namespace fs = std::filesystem;

inline bool
write_magic_bishops(std::ofstream& out) {
    using namespace movegen::detail;
    using bishop_t = MagicsBase<BISHOP, ShiftIndexer>;
    const auto b = std::make_unique<bishop_t>();
    b->init();
    std::vector<uint8_t> bytes;
    b->write(std::back_inserter(bytes));
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out.good();
}

inline bool
write_magic_bishops_pext(std::ofstream& out) {
    using namespace movegen::detail;
    using bishop_t = MagicsBase<BISHOP, PEXTIndexer>;
    const auto b = std::make_unique<bishop_t>();
    b->init();
    std::vector<uint8_t> bytes;
    b->write(std::back_inserter(bytes));
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out.good();
}


inline bool
write_magic_rooks_pext(std::ofstream& out) {
    using namespace movegen::detail;
    auto b = std::make_unique<MagicsBase<ROOK, PEXTIndexer>>();
    b->init();
    std::vector<uint8_t> bytes;
    b->write(std::back_inserter(bytes));
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out.good();
}

inline bool
write_magic_rooks(std::ofstream& out) {
    using namespace movegen::detail;
    auto b = std::make_unique<MagicsBase<ROOK, ShiftIndexer>>();
    b->init();
    std::vector<uint8_t> bytes;
    b->write(std::back_inserter(bytes));
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out.good();
}

inline bool
write_lines(std::ofstream& out) {
    auto l = std::make_unique<movegen::detail::lines_type>(std::in_place, movegen::detail::compute_lines);
    std::vector<uint8_t> bytes;
    utils::write_range(*l, std::back_inserter(bytes));
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out.good();
}

inline bool
write_from_to(std::ofstream& out) {
    auto l = std::make_unique<movegen::detail::lines_type>(std::in_place, movegen::detail::compute_from_to);
    std::vector<uint8_t> bytes;
    utils::write_range(*l, std::back_inserter(bytes));
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out.good();
}

int
main(int argc, char** argv) {
    std::unordered_map<std::string, std::function<bool(std::ofstream&)>> targets;
    targets.emplace("magic_rooks", write_magic_rooks);
    targets.emplace("magic_bishops", write_magic_bishops);
    targets.emplace("lines", write_lines);
    targets.emplace("from_to", write_from_to);
    targets.emplace("magic_rooks_pext", write_magic_rooks_pext);
    targets.emplace("magic_bishops_pext", write_magic_bishops_pext);

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
        if (!out) {
            throw std::runtime_error(fmt::format("failed to open file {}", path.string()));
        }
        if (!targets.at(target)(out)) {
            throw std::runtime_error("Failed to write target data");
        }
    } catch (const std::exception& e) {
        fmt::println(stderr, "{}", e.what());
        fs::remove(path);
        return 1;
    }

    return 0;
}