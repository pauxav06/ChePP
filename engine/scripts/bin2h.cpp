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
    if(argc!=5){
        std::cerr<<"Usage: "<<argv[0]<<" <raw.bin> <config.txt> <output.h> <output.cpp>\n";
        return 1;
    }

    std::ifstream cfg(argv[2]);
    std::ifstream raw(argv[1],std::ios::binary);
    std::ofstream h(argv[3]);
    std::ofstream cpp(argv[4]);

    if(!cfg || !raw || !h || !cpp){ std::cerr<<"Failed to open files\n"; return 1; }

    h << "#pragma once\n#include <cstdint>\n\n";
    cpp << "#include \"" << argv[3] << "\"\n\n";

    std::vector<std::unique_ptr<LayerBase>> layers;
    std::string type_str, name;
    uint64_t size;

    while(cfg>>type_str>>size>>name){
        LayerType t = parse_type(type_str);
        layers.push_back(make_layer(t,size,name));
    }

    for(auto &layer : layers){
        layer->emit_declaration(h);

        size_t bytes = layer->type_size() * layer->size;
        std::vector<uint8_t> buf(bytes);
        raw.read(reinterpret_cast<char*>(buf.data()), bytes);
        if(!raw){ std::cerr<<"Raw file too small\n"; return 1; }

        layer->emit_definition(cpp, buf.data());
    }

    std::cout<<"Embedded all layers into "<<argv[3]<<" and "<<argv[4]<<"\n";
}
