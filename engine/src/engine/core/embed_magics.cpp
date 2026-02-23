#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "bitboard.h"
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

using namespace chepp;
namespace fs = std::filesystem;
using json   = nlohmann::json;

int
main(int argc, char** argv) {
    argparse::ArgumentParser program("embed_bitboards");
    program.add_argument("--bin-dir").required().help("Directory to output binary files");
    program.add_argument("--json").required().help("Path to JSON output file");
    program.parse_args(argc, argv);

    fs::path bin_dir   = program.get<std::string>("--bin-dir");
    fs::path json_path = program.get<std::string>("--json");

    fs::create_directories(bin_dir);

    json manifest{};

    try {

        {
            fs::path bishop_file = bin_dir / "bishop.bin";
            auto     b           = std::make_unique<movegen::detail::Magics<BISHOP>>();
            b->init();
            std::ofstream     out(bishop_file, std::ios::binary | std::ios::trunc);
            std::vector<char> data(sizeof(*b));
            std::memcpy(data.data(), b.get(), sizeof(*b));
            out.write(data.data(), data.size());
            manifest["bishop"] = bishop_file.string();
        }

        {
            fs::path rook_file = bin_dir / "rook.bin";
            auto     r         = std::make_unique<movegen::detail::Magics<ROOK>>();
            r->init();
            std::ofstream     out(rook_file, std::ios::binary | std::ios::trunc);
            std::vector<char> data(sizeof(*r));
            std::memcpy(data.data(), r.get(), sizeof(*r));
            out.write(data.data(), data.size());
            manifest["rook"] = rook_file.string();
        }

        auto     l = std::make_unique<movegen::detail::lines_type>(std::in_place, movegen::detail::compute_lines);
        fs::path lines_file = bin_dir / "lines.bin";
        {
            std::ofstream     out(lines_file, std::ios::binary | std::ios::trunc);
            std::vector<char> data(sizeof(*l));
            std::memcpy(data.data(), l.get(), sizeof(*l));
            out.write(data.data(), data.size());
        }
        manifest["lines"] = lines_file.string();

        auto     f = std::make_unique<movegen::detail::from_to_type>(std::in_place, movegen::detail::compute_from_to);
        fs::path from_to_file = bin_dir / "from_to.bin";
        {
            std::ofstream     out(from_to_file, std::ios::binary | std::ios::trunc);
            std::vector<char> data(sizeof(*f));
            std::memcpy(data.data(), f.get(), sizeof(*f));
            out.write(data.data(), data.size());
        }
        manifest["from_to"] = from_to_file.string();

        {
            std::ofstream json_out(json_path);
            json_out << manifest.dump(4);
        }

        std::cout << "All files written successfully to " << bin_dir << "\n";
        std::cout << "Manifest JSON: " << json_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        fs::remove_all(bin_dir);
        return 1;
    }

    return 0;
}