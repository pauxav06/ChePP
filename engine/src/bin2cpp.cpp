#include <argparse/argparse.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

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

    program.parse_args(argc, argv);

    fs::path input_path = program.get<std::string>("--input");
    auto     name       = program.get<std::string>("--name");
    fs::path output_dir = program.get<std::string>("--output_dir");

    auto header_path = (output_dir / name) += ".h";
    auto inc_path    = (output_dir / name) += ".inc";

    try {
        std::ifstream input(input_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(std::format("Failed to open input file {}", input_path.string()));
        }
        std::ofstream header(header_path, std::ios::binary);
        if (!header) {
            throw std::runtime_error(std::format("Failed to open header {}", header_path.string()));
        }
        std::ofstream inc(inc_path, std::ios::binary);
        if (!inc) {
            throw std::runtime_error(std::format("Failed to open inc {}", inc_path.string()));
        }

        std::vector<uint8_t> data((std::istreambuf_iterator(input)), std::istreambuf_iterator<char>());

        std::string array_name = "GENERATED_" + sanitize_name(name);

        std::print(header, "#pragma once\n");
        std::print(header, "#include <array>\n");
        std::print(header, "#include <cstdint>\n\n");
        std::print(header, "#if defined(IDE) && IDE == 1\n");
        std::print(header, "inline constexpr std::array<uint8_t, {}> {}{{}};\n", data.size(), array_name);
        std::print(header, "#else\n");
        std::print(header, "#include \"{}\"\n", inc_path.string());
        std::print(header, "#endif\n");

        std::print(inc, "inline constexpr std::array<uint8_t, {}> {}{{\n", data.size(), array_name);
        for (std::size_t i = 0; i < data.size(); ++i) {
            std::print(inc, "0x{:02X}", data[i]);
            if (i != (data.size() - 1)) {
                std::print(inc, ",");
            }
            if ((i + 1) % 16 == 0) std::print(inc, "\n");
        }
        std::print(inc, "\n}};\n");

        std::print(std::cout, "Generated {} ({} bytes)\n", header_path.string(), data.size());
    } catch (std::exception& e) {
        fs::remove(input_path);
        fs::remove(inc_path);
        fs::remove(header_path);
        std::print(std::cerr, "Error: {}\n", e.what());
        return 1;
    }
}