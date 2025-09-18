//
// Created by paul on 9/17/25.
//

#include <iostream>
#include <cstdlib>
#include "data_loader.h"
#include "converter/grapheus_converter.h"
#include "stream/binpack_sfen_input_stream.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

void binpack2grapheus(
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& train_out,
    const std::vector<std::string>& val_out, const float val_ratio
) {
    auto make_binpack_stream = [](const std::string& filename)
    {
        constexpr DataloaderSkipConfig skip_config{true, 0, false, true, 0, 1};
        return FilteredBinpackSfenInputStream(
            filename,
            false,
            make_skip_predicate(skip_config)
        );
    };
    binpack_convert<binpack::TrainingDataEntry, GrapheusData::Header, GrapheusData::Position, FilteredBinpackSfenInputStream>(
        inputs,
        train_out,
        val_out,
        make_binpack_stream,
        GrapheusData::make_header,
        GrapheusData::Position::from_binpack_entry,
        val_ratio
    );
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Failed to open config file: " << argv[1] << "\n";
        return 1;
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << "\n";
        return 1;
    }

    std::vector<std::string> inputs;
    if (j.contains("input_dir")) {
        if (const fs::path dir(j["input_dir"].get<std::string>()); !dir.empty()) {
            if (!fs::exists(dir) || !fs::is_directory(dir)) {
                std::cerr << "Input directory invalid: " << dir << "\n";
                return 1;
            }
            for (auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_regular_file()) {
                    inputs.push_back(entry.path().string());
                }
            }
        }
    }
    if (j.contains("inputs")) {
        auto vec = j["inputs"].get<std::vector<std::string>>();
        inputs.insert(inputs.end(), vec.begin(), vec.end());
    }
    if (inputs.empty()) {
        std::cerr << "No input files provided\n";
        return 1;
    }
    for (const auto& f : inputs) {
        if (!fs::exists(f)) {
            std::cerr << "Input file missing: " << f << "\n";
            return 1;
        }
    }

    std::vector<std::string> train_out;
    std::vector<std::string> val_out;


    if (!j.contains("n_threads")) {
        std::cerr << "Must specify number of threads\n";
        return 1;
    }
    std::size_t n_threads = j["n_threads"].get<size_t>();


    if (!j.contains("train_out_dir") && !j.contains("val_out_dir")) {
        std::cerr << "Must specify train_out_dir & val_out_dir\n";
        return 1;
    }
    fs::path train_dir(j["train_out_dir"].get<std::string>());
    fs::path val_dir(j["val_out_dir"].get<std::string>());

    if (!fs::exists(train_dir)) fs::create_directories(train_dir);
    if (!fs::exists(val_dir)) fs::create_directories(val_dir);

    for (size_t i = 0; i < n_threads; i++) {
        train_out.push_back((train_dir / std::format("train_{}.bin", i)).string());
        val_out.push_back((val_dir / std::format("val_{}.bin", i)).string());
    }

    float val_ratio      = j.value("val_ratio", 0.1f);

    if (val_ratio < 0.0f || val_ratio > 1.0f) {
        std::cerr << "Invalid parameters: 0 <= val_ratio <= 1\n";
        return 1;
    }

    std::cout << "Starting conversion with " << inputs.size() << " input file(s) and " << n_threads << " n threads(s)\n";
    std::cout << "outputting " << n_threads << " files to " << train_dir << " and " << val_dir << "\n";
    std::cout << "val ratio: " << val_ratio << std::endl;
    binpack2grapheus(inputs, train_out, val_out, val_ratio);
    return 0;
}