#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include "argparse.hpp"

#include <format>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

enum class LayerType : uint8_t {
    UINT8=1, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, FLOAT, DOUBLE
};

struct LayerBase {
    uint64_t size;
    std::string name;

    LayerBase(uint64_t sz, std::string n) : size(sz), name(std::move(n)) {}
    virtual ~LayerBase() = default;

    [[nodiscard]] virtual std::string cpp_type() const = 0;
    [[nodiscard]] virtual std::string literal_suffix() const = 0;
    [[nodiscard]] virtual size_t type_size() const = 0;

    void emit_declaration(std::ofstream &h) const {
        h << "alignas(64) extern " << cpp_type() << " " << name << "[" << size << "];\n";
    }

    virtual void emit_definition(std::ofstream &cpp, const uint8_t* data) const = 0;
};

template<typename T>
struct LayerImpl final : LayerBase {
    LayerImpl(uint64_t sz, std::string n) : LayerBase(sz,std::move(n)) {}

    [[nodiscard]] std::string cpp_type() const override {
        if constexpr(std::is_same_v<T,uint8_t>) return "uint8_t";
        else if constexpr(std::is_same_v<T,int8_t>) return "int8_t";
        else if constexpr(std::is_same_v<T,uint16_t>) return "uint16_t";
        else if constexpr(std::is_same_v<T,int16_t>) return "int16_t";
        else if constexpr(std::is_same_v<T,uint32_t>) return "uint32_t";
        else if constexpr(std::is_same_v<T,int32_t>) return "int32_t";
        else if constexpr(std::is_same_v<T,uint64_t>) return "uint64_t";
        else if constexpr(std::is_same_v<T,int64_t>) return "int64_t";
        else if constexpr(std::is_same_v<T,float>) return "float";
        else if constexpr(std::is_same_v<T,double>) return "double";
    }

    [[nodiscard]] std::string literal_suffix() const override {
        if constexpr(std::is_same_v<T,uint32_t>) return "UL";
        else if constexpr(std::is_same_v<T,int32_t>) return "L";
        else if constexpr(std::is_same_v<T,uint64_t>) return "ULL";
        else if constexpr(std::is_same_v<T,int64_t>) return "LL";
        else if constexpr(std::is_same_v<T,float>) return "f";
        else if constexpr(std::is_same_v<T,double>) return "d";
        else return "";
    }

    size_t type_size() const override { return sizeof(T); }

    void emit_definition(std::ofstream& cpp, const uint8_t* data) const override {
        cpp << "alignas(64) " << cpp_type() << " " << name << "[" << size << "] = {";
        const T* arr = reinterpret_cast<const T*>(data);
        for(uint64_t i=0;i<size;i++){
            if(i%8==0) cpp << "\n  ";
            if constexpr(std::is_same_v<T,uint8_t> || std::is_same_v<T,int8_t>) {
                cpp << +arr[i];
            } else {
                cpp << arr[i] << literal_suffix();
            }
            cpp << ", ";
        }
        cpp << "\n};\n\n";
    }
};

LayerType parse_type(const std::string &s) {
    static std::unordered_map<std::string,LayerType> m = {
        {"uint8", LayerType::UINT8}, {"int8", LayerType::INT8},
        {"uint16",LayerType::UINT16}, {"int16",LayerType::INT16},
        {"uint32",LayerType::UINT32}, {"int32",LayerType::INT32},
        {"uint64",LayerType::UINT64}, {"int64",LayerType::INT64},
        {"float", LayerType::FLOAT}, {"double", LayerType::DOUBLE}
    };
    const auto it = m.find(s);
    if(it==m.end()) throw std::runtime_error("Unknown type: "+s);
    return it->second;
}

std::unique_ptr<LayerBase> make_layer(const LayerType t, uint64_t size, const std::string& name) {
    switch(t){
        case LayerType::UINT8:  return std::make_unique<LayerImpl<uint8_t>>(size,name);
        case LayerType::INT8:   return std::make_unique<LayerImpl<int8_t>>(size,name);
        case LayerType::UINT16: return std::make_unique<LayerImpl<uint16_t>>(size,name);
        case LayerType::INT16:  return std::make_unique<LayerImpl<int16_t>>(size,name);
        case LayerType::UINT32: return std::make_unique<LayerImpl<uint32_t>>(size,name);
        case LayerType::INT32:  return std::make_unique<LayerImpl<int32_t>>(size,name);
        case LayerType::UINT64: return std::make_unique<LayerImpl<uint64_t>>(size,name);
        case LayerType::INT64:  return std::make_unique<LayerImpl<int64_t>>(size,name);
        case LayerType::FLOAT:  return std::make_unique<LayerImpl<float>>(size,name);
        case LayerType::DOUBLE: return std::make_unique<LayerImpl<double>>(size,name);
    }
    throw std::runtime_error("Invalid LayerType");
}

int main(int argc,char**argv){
    fs::path header_file{};
    fs::path cpp_file{};
    try {
        argparse::ArgumentParser parser("bin2h");
        parser.add_argument("--raw").required().help("Input raw binary file");
        parser.add_argument("--config").required().help("JSON config file");
        parser.add_argument("--header").required().help("Output header file");
        parser.add_argument("--cpp").required().help("Output cpp file");

        try {
            parser.parse_args(argc, argv);
        } catch (const std::exception& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << parser;
            std::exit(1);
        }

        fs::path raw_file = parser.get("--raw");
        fs::path cfg_file = parser.get("--config");
        header_file = parser.get("--header");
        cpp_file = parser.get("--cpp");

        std::ifstream raw(raw_file, std::ios::binary);
        if(!raw) throw std::runtime_error("Failed to open raw file: " + raw_file.string());

        std::ifstream cfg(cfg_file);
        if(!cfg) throw std::runtime_error("Failed to open config file: " + cfg_file.string());
        json j; cfg >> j;

        std::ofstream h(header_file ,std::ofstream::trunc);
        std::ofstream cpp(cpp_file, std::ofstream::trunc);
        if(!h || !cpp) throw std::runtime_error("Failed to open output files");

        h << "#pragma once\n#include <cstdint>\n\n";
        cpp << "#include " << absolute(header_file) << "\n\n";

        std::vector<std::unique_ptr<LayerBase>> layers;
        for(const auto& entry : j){
            std::string type_str = entry.at("type");
            uint64_t size = entry.at("size");
            std::string name = entry.at("name");
            LayerType t = parse_type(type_str);
            layers.push_back(make_layer(t,size,name));
        }

        size_t total_size = 0;
        for(auto &layer : layers){
            layer->emit_declaration(h);

            size_t bytes = layer->type_size() * layer->size;
            total_size += bytes;
            std::vector<uint8_t> buf(bytes);
            raw.read(reinterpret_cast<char*>(buf.data()), bytes);
            if(static_cast<size_t>(raw.gcount()) != bytes){
                throw std::runtime_error(std::format("Raw file too small: expected a layer to be {} bytes and {} bytes were left"
                    , std::to_string(bytes), std::to_string(raw.gcount())));
            }

            layer->emit_definition(cpp, buf.data());
        }

        if (raw.tellg() != fs::file_size(raw_file.string()))
        {
            throw std::runtime_error(std::format("Raw file too big: read {} out of {} bytes", total_size, fs::file_size(raw_file.string())));
        }

        std::cout<<"Embedded all layers into "<<header_file<<" and "<<cpp_file<<"\n";
        return 0;
    } catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<"\n";
        if(argc >= 4) {
            if (!header_file.empty()) fs::remove(header_file);
            if (!cpp_file.empty()) fs::remove(cpp_file);
        }
        return 1;
    }
}
