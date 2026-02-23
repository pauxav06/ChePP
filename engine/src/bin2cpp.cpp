#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

std::string
sanitize_name(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (isalnum(c)) {
            result += std::toupper(c);
        } else {
            result += '_';
        }
    }
    return result;
}

int
main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: bin2cpp <manifest.json> <output_dir>\n";
        return 1;
    }

    fs::path manifest_path = argv[1];
    fs::path out_dir       = argv[2];
    fs::create_directories(out_dir);

    std::ifstream manifest_file(manifest_path);
    if (!manifest_file) {
        std::cerr << "Failed to open manifest file\n";
        return 1;
    }

    json manifest{};
    manifest_file >> manifest;

    for (auto& [name, path_str] : manifest.items()) {
        fs::path      bin_path = path_str.get<std::string>();
        std::ifstream bin_file(bin_path, std::ios::binary);
        if (!bin_file) {
            std::cerr << "Failed to open " << bin_path << "\n";
            continue;
        }

        std::vector<uint8_t> data((std::istreambuf_iterator<char>(bin_file)), std::istreambuf_iterator<char>());

        std::string array_name  = "GENERATED_" + sanitize_name(name);
        fs::path    header_path = out_dir / (name + ".h");

        std::ofstream out(header_path);
        if (!out) {
            std::cerr << "Failed to open output file " << header_path << "\n";
            continue;
        }

        out << "#pragma once\n\n";
        out << "#include <array>\n";
        out << "#include <cstdint>\n\n";
        out << "static constexpr std::array<uint8_t, " << data.size() << "> " << array_name << " = {\n    ";

        for (size_t i = 0; i < data.size(); ++i) {
            out << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
            if (i + 1 != data.size()) out << ", ";
            if ((i + 1) % 12 == 0) out << "\n    ";
        }

        out << "\n};\n";

        std::cout << "Generated " << header_path << " (" << data.size() << " bytes)\n";
    }

    return 0;
}