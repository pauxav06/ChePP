#include "format.h"

#include <argparse/argparse.hpp>
#include <cassert>
#include <filesystem>
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

    auto header_path = (output_dir / name) += ".h";
    auto inc_path    = (output_dir / name) += ".inc";

    try {
        file_ptr input{fopen(input_path.string().c_str(), "rb")};
        if (!input.get()) {
            throw std::runtime_error(fmt::format("Failed to open input file {}", input_path.string()));
        }
        file_ptr header{fopen(header_path.string().c_str(), "wb")};
        if (!header.get()) {
            throw std::runtime_error(fmt::format("Failed to open header {}", header_path.string()));
        }
        file_ptr inc{fopen(inc_path.string().c_str(), "wb")};
        if (!inc.get()) {
            throw std::runtime_error(fmt::format("Failed to open inc {}", inc_path.string()));
        }

        std::vector<uint8_t> data(fs::file_size(input_path));
        std::rewind(input.get());
        auto bytes = std::fread(std::data(data), 1, std::size(data), input.get());
        if (bytes != std::size(data)) {
            if (std::feof(input.get())) {
                fmt::println(stdout, "eof reched", bytes);
            } else if (std::ferror(inc.get())) {
                std::perror(nullptr);
            }
            throw std::runtime_error(
                fmt::format("Failed to read bytes in {} ({}/{})", input_path.string(), bytes, std::size(data)));
        }

        std::string array_name = "GENERATED_" + sanitize_name(name);

        fmt::println(header.get(), "#pragma once");
        fmt::println(header.get(), "#include <array>");
        fmt::println(header.get(), "#include <cstdint>");
        fmt::println(header.get(), "");
        fmt::println(header.get(), "#if defined(IDE) && IDE == 1");
        fmt::println(
            header.get(), "inline {} const std::array<uint8_t, {}> {}{{}};", qualifier, data.size(), array_name);
        fmt::println(header.get(), "#else");
        fmt::println(header.get(), "#include \"{}\"", inc_path.string());
        fmt::println(header.get(), "#endif");

        fmt::println(inc.get(), "inline {} const std::array<uint8_t, {}> {}{{", qualifier, data.size(), array_name);
        for (std::size_t i = 0; i < data.size(); ++i) {
            fmt::print(inc.get(), "0x{:02X}", data[i]);
            if (i != (data.size() - 1)) {
                fmt::print(inc.get(), ",");
            }
            if ((i + 1) % 16 == 0) fmt::println(inc.get(), "");
        }
        fmt::println(inc.get(), "\n}};");

        fmt::println(stdout, "Generated {} ({} bytes)", header_path.string(), data.size());
    } catch (std::exception& e) {
        fs::remove(input_path);
        fs::remove(inc_path);
        fs::remove(header_path);
        fmt::println(stdout, "Error: {}", e.what());
        return 1;
    }
}