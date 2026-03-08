#include "format.h"

#include <argparse/argparse.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

inline std::string
sanitize_name(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (isalnum(c)) {
            result += static_cast<char>(std::toupper(c));
        } else {
            result += '_';
        }
    }
    assert(!result.empty());
    return result;
}

int
main(int argc, char** argv) {
    argparse::ArgumentParser program("bin2cpp");
    program.add_argument("--input").required().help("Path to input file");
    program.add_argument("--output_dir").required().help("Path to output directory");
    program.add_argument("--name").required().help("Name");
    program.add_argument("--q").choices("", "constexpr", "constinit").default_value("");

    program.parse_args(argc, argv);

    fs::path input_path = program.get<std::string>("--input");
    auto     name       = program.get<std::string>("--name");
    fs::path output_dir = program.get<std::string>("--output_dir");
    auto     qualifier  = program.get<std::string>("--q");

    auto header_path = output_dir / (name + ".h");
    auto inc_path    = output_dir / (name + ".inc");

    try {
        std::ifstream input(input_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(fmt::format("Failed to open input file {}", input_path.string()));
        }

        std::ofstream header(header_path, std::ios::binary);
        if (!header) {
            throw std::runtime_error(fmt::format("Failed to open header {}", header_path.string()));
        }

        std::ofstream inc(inc_path, std::ios::binary);
        if (!inc) {
            throw std::runtime_error(fmt::format("Failed to open inc {}", inc_path.string()));
        }

        std::vector<uint8_t> data(fs::file_size(input_path));
        input.read(reinterpret_cast<char*>(data.data()), data.size());
        if (!input) {
            throw std::runtime_error(fmt::format("Failed to read bytes in {} ({} bytes read, expected {})",
                                                 input_path.string(),
                                                 input.gcount(),
                                                 data.size()));
        }

        std::string array_name = "GENERATED_" + sanitize_name(name);

        fmt::println(header, "#pragma once");
        fmt::println(header, "#include <array>");
        fmt::println(header, "#include <cstdint>");
        fmt::println(header, "");
        fmt::println(header, "#if defined(IDE) && IDE == 1");
        fmt::println(header, "inline {} const std::array<uint8_t, {}> {}{{}};", qualifier, data.size(), array_name);
        fmt::println(header, "#else");
        fmt::println(header, "#include \"{}\"", inc_path.string());
        fmt::println(header, "#endif");

        fmt::println(inc, "inline {} const std::array<uint8_t, {}> {}{{", qualifier, data.size(), array_name);
        for (std::size_t i = 0; i < data.size(); ++i) {
            fmt::print(inc, "0x{:02X}", data[i]);
            if (i != data.size() - 1) {
                fmt::print(inc, ",");
            }
            if ((i + 1) % 16 == 0) {
                fmt::println(inc, "");
            }
        }
        fmt::println(inc, "\n}};");

        fmt::println(stdout, "Generated {} ({} bytes)", header_path.string(), data.size());
    } catch (const std::exception& e) {
        fs::remove(input_path);
        fs::remove(inc_path);
        fs::remove(header_path);
        fmt::println(stdout, "Error: {}", e.what());
        return 1;
    }

    return 0;
}