#include <argparse/argparse.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json   = nlohmann::json;
namespace fs = std::filesystem;

template <typename T>
std::string
litteral_suffix() {
    return "";
}
template <>
std::string
litteral_suffix<float>() {
    return "f";
}
template <>
std::string
litteral_suffix<double>() {
    return "";
}
template <>
std::string
litteral_suffix<uint8_t>() {
    return "U";
}
template <>
std::string
litteral_suffix<uint16_t>() {
    return "U";
}
template <>
std::string
litteral_suffix<uint32_t>() {
    return "U";
}
template <>
std::string
litteral_suffix<uint64_t>() {
    return "ULL";
}

template <typename T>
struct type_name {
    static std::string
    name() {
        return "unknown";
    }
};
template <>
struct type_name<float> {
    static std::string
    name() {
        return "float";
    }
};
template <>
struct type_name<double> {
    static std::string
    name() {
        return "double";
    }
};
template <>
struct type_name<int8_t> {
    static std::string
    name() {
        return "int8_t";
    }
};
template <>
struct type_name<uint8_t> {
    static std::string
    name() {
        return "uint8_t";
    }
};
template <>
struct type_name<int16_t> {
    static std::string
    name() {
        return "int16_t";
    }
};
template <>
struct type_name<uint16_t> {
    static std::string
    name() {
        return "uint16_t";
    }
};
template <>
struct type_name<int32_t> {
    static std::string
    name() {
        return "int32_t";
    }
};
template <>
struct type_name<uint32_t> {
    static std::string
    name() {
        return "uint32_t";
    }
};
template <>
struct type_name<int64_t> {
    static std::string
    name() {
        return "int64_t";
    }
};
template <>
struct type_name<uint64_t> {
    static std::string
    name() {
        return "uint64_t";
    }
};

template <typename Func>
void
dispatch_type(const std::string& type_str, Func&& func) {
    if (type_str == "float")
        func.template operator()<float>();
    else if (type_str == "double")
        func.template operator()<double>();
    else if (type_str == "int8")
        func.template operator()<int8_t>();
    else if (type_str == "uint8")
        func.template operator()<uint8_t>();
    else if (type_str == "int16")
        func.template operator()<int16_t>();
    else if (type_str == "uint16")
        func.template operator()<uint16_t>();
    else if (type_str == "int32")
        func.template operator()<int32_t>();
    else if (type_str == "uint32")
        func.template operator()<uint32_t>();
    else if (type_str == "int64")
        func.template operator()<int64_t>();
    else if (type_str == "uint64")
        func.template operator()<uint64_t>();
    else
        throw std::runtime_error("Unsupported dtype: " + type_str);
}

template <typename T>
void
write_header(std::ofstream& out_h, const std::string& name, size_t reps, size_t flat_size) {
    out_h << "alignas(64) extern const " << type_name<T>::name() << " " << name << "[" << reps << "][" << flat_size
          << "];\n";
}

template <typename T>
void
write_cpp(std::ofstream& out_cpp, const std::string& name, const std::vector<std::vector<T>>& data) {
    size_t reps      = data.size();
    size_t flat_size = data[0].size();

    out_cpp << "alignas(64) constexpr " << type_name<T>::name() << " " << name << "[" << reps << "][" << flat_size
            << "] = {\n";

    for (size_t r = 0; r < reps; ++r) {
        out_cpp << "    {";
        for (size_t i = 0; i < flat_size; ++i) {
            out_cpp << +data[r][i] << litteral_suffix<T>();
            if (i + 1 < flat_size) out_cpp << ", ";
        }
        out_cpp << "}";
        if (r + 1 < reps) out_cpp << ",\n";
    }
    out_cpp << "\n};\n";
}

template <typename T>
void
process_layer(std::ifstream& bin, const json& layer, std::ofstream& out_h, std::ofstream& out_cpp) {
    std::string name = layer["name"];
    size_t      rows = layer["rows"];
    size_t      cols = layer["cols"];
    size_t      rep  = layer.value("repetition", 1);

    std::vector<std::vector<T>> all_reps;

    for (size_t r = 0; r < rep; ++r) {
        size_t         count = rows * cols;
        std::vector<T> data(count);
        bin.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
        all_reps.push_back(std::move(data)); // store flattened
    }

    write_header<T>(out_h, name, rep, rows * cols);
    write_cpp<T>(out_cpp, name, all_reps);
}

int
main(int argc, char** argv) {
    argparse::ArgumentParser program("bin2cpp");
    program.add_argument("-i", "--input").required().help("Input binary file");
    program.add_argument("--header").required().help("Output header name");
    program.add_argument("--cpp").required().help("Output cpp name");
    program.add_argument("-c", "--config").required().help("JSON config file");
    program.parse_args(argc, argv);

    fs::path input_file  = program.get<std::string>("--input");
    fs::path out_h       = program.get<std::string>("--header");
    fs::path out_cpp     = program.get<std::string>("--cpp");
    fs::path config_file = program.get<std::string>("--config");

    json cfg;
    {
        std::ifstream f(config_file);
        f >> cfg;
    }

    std::ofstream header_file(out_h, std::ofstream::binary | std::ofstream::trunc);
    std::ofstream cpp_file(out_cpp, std::ofstream::binary | std::ofstream::trunc);

    if (!header_file || !cpp_file) throw std::runtime_error("Failed to open output files");

    header_file << "#pragma once\n#include <cstddef>\n#include <cstdint>\n\n";
    cpp_file << "#include \"" << fs::absolute(out_h).string() << "\"\n\n";

    std::ifstream bin(input_file, std::ios::binary | std::ios::ate);
    if (!bin) {
        std::cerr << "Failed to open binary file.\n";
        return 1;
    }
    std::streamsize total_size = bin.tellg();
    bin.seekg(0, std::ios::beg);

    std::streamsize bytes_read = 0;

    try {
        for (const auto& layer : cfg["layers"]) {
            std::string type_str = layer.value("dtype", "float");
            dispatch_type(type_str, [&]<typename T>() {
                auto pos_before = bin.tellg();
                process_layer<T>(bin, layer, header_file, cpp_file);
                auto pos_after = bin.tellg();
                bytes_read += (pos_after - pos_before);
            });
        }

        if (bytes_read != total_size) {
            std::cerr << "Binary file was not fully consumed: expected " << total_size << " bytes, read " << bytes_read
                      << " bytes.\n";
            header_file.close();
            cpp_file.close();
            std::filesystem::remove(out_h);
            std::filesystem::remove(out_cpp);
            return 1;
        }

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
