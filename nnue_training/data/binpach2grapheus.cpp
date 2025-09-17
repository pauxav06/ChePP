//
// Created by paul on 9/17/25.
//

#include <iostream>
#include <cstdlib>
#include "data_loader.h"
#include "converter/grapheus_converter.h"
#include "binpack/nnue_training_data_stream.h"
#include "stream/binpack_sfen_input_stream.h"
#include "stream/stream_view.h"

void binpack2grapheus(const std::string& input, const std::string& train, const std::string& val, const size_t n,
                      const int chunk_size,
                      const float val_ = 0.1f) {
    constexpr DataloaderSkipConfig skip_config {true, 0, false, true, 0, 1};
    FilteredBinpackSfenInputStream fs(input, false, n, make_skip_predicate(skip_config));
    StreamView sv(fs);

    using OutT = GrapheusData::Position;

    auto [train_files, val_files] = convert_and_shuffle_chunks(sv, grapheus_converter, chunk_size, val_);

    auto write_header_and_data = [] (std::vector<TmpFile>& tmp, const std::string& file, const std::string& message = "Amazing dataset nb 96") {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        OutT::Header header {};
        std::memcpy(header.label_1, message.c_str(), std::min(message.length() + 1, sizeof(header.label_1)));
        header.entry_count = count_all_elements<OutT>(tmp);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        merge_and_write<OutT>(tmp, out);
    };

    write_header_and_data(train_files, train);
    write_header_and_data(val_files, val);
}

int main(const int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_binpack> <train_out> <val_out> <num_positions> <chunk_size> [val_ratio]\n";
        return 1;
    }

    const char* input_path = argv[1];
    const char* train_path = argv[2];
    const char* val_path = argv[3];
    const size_t num_positions = std::stoull(argv[4]);
    const int chunk_size = std::stoi(argv[5]);
    const float val_ratio = (argc >= 7) ? std::stof(argv[6]) : 0.1f;

    binpack2grapheus(input_path, train_path, val_path, num_positions, chunk_size, val_ratio);

    return 0;
}
