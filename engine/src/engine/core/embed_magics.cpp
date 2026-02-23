#include <filesystem>
#include <fstream>
#include <iostream>

#include "bitboard.h"
#include <argparse/argparse.hpp>

using namespace chepp;
namespace fs = std::filesystem;

int
main(int argc, char** argv) {
    argparse::ArgumentParser program("bin2cpp");
    program.add_argument("--header").required().help("Output header name");
    program.add_argument("--cpp").required().help("Output cpp name");
    program.parse_args(argc, argv);

    fs::path out_h   = program.get<std::string>("--header");
    fs::path out_cpp = program.get<std::string>("--cpp");

    std::ofstream header_file(out_h, std::ofstream::binary | std::ofstream::trunc);
    header_file << "#pragma once\n";
    std::ofstream cpp_file(out_cpp, std::ofstream::binary | std::ofstream::trunc);

    if (!header_file || !cpp_file) throw std::runtime_error("Failed to open output files");

    try {
        auto b = std::make_unique<movegen::detail::Magics<BISHOP>>();
        b->init();
        header_file << b->write_declaration("BISHOP") << "\n";
        cpp_file << b->write_cpp(out_h.string(), "BISHOP") << "\n";

        auto r = std::make_unique<movegen::detail::Magics<ROOK>>();
        r->init();
        header_file << r->write_declaration("ROOK") << "\n";
        cpp_file << r->write_cpp(out_h.string(), "ROOK") << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        header_file.close();
        cpp_file.close();
        std::filesystem::remove(out_h);
        std::filesystem::remove(out_cpp);
        return 1;
    }

    std::cout << "Files generated successfully:\n"
              << "  Header: " << out_h << "\n"
              << "  CPP: " << out_cpp << "\n";

    return 0;
}
