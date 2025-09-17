//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_DATA_LOADER_H
#define CHEPP_DATA_LOADER_H

#include "utils/utils.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <ranges>
#include <vector>

template <typename ElemT>
TmpFile process_chunk(std::vector<ElemT> chunk) {
    TmpFile      tmp;
    auto& ofs = tmp.ofstream();

    std::ranges::shuffle(chunk, rng::get_thread_local_rng());
    std::ranges::for_each(chunk, [&ofs](auto i) {ofs.write(reinterpret_cast<char*>(&i), sizeof(i));});
    ofs.flush();

    std::cout << "Wrote " << chunk.size() << " chunks to" << tmp.path() << std::endl;

    return tmp;
}


template <std::ranges::input_range Stream, typename Converter>
requires std::invocable<Converter, std::ranges::range_value_t<Stream>>
auto convert_and_shuffle_chunks(
    Stream& sv,
    Converter conv,
    size_t chunk_size,
    const double val_split = 0.1
) {
    using input_t  = std::ranges::range_value_t<Stream>;
    using output_t = std::invoke_result_t<Converter, input_t>;
    using chunk_t  = std::vector<output_t>;

    std::vector<std::future<TmpFile>> futures{};
    auto view_chunk = sv | std::views::transform(conv) | std::views::chunk(chunk_size);
    auto to_future = [&] (auto&& view) {
        chunk_t chunk;
        chunk.reserve(chunk_size);
        std::ranges::copy(view, std::back_inserter(chunk));
        return std::async(std::launch::async, process_chunk<output_t>, std::move(chunk));
    };
    std::ranges::transform(view_chunk, std::back_inserter(futures), to_future);

    std::vector<TmpFile> train_files, val_files;
    for (auto& fut : futures) {
        (std::bernoulli_distribution(1.0 - val_split)(rng::get_thread_local_rng()) ?
        train_files : val_files).push_back(fut.get());
    }

    return std::pair{std::move(train_files), std::move(val_files)};
}


template<typename ElemT>
std::vector<size_t> count_elements(std::vector<TmpFile>& temp_files) {
    auto count = [] (TmpFile& file) {
        auto& stream = file.ifstream();
        stream.seekg(0, std::ios::end);
        const size_t c = stream.tellg() / static_cast<long>(sizeof(ElemT));
        stream.seekg(0);
        return c;
    };
    auto counts = temp_files | std::views::transform(count);
    return std::vector(counts.begin(), counts.end());
}

template<typename ElemT>
size_t count_all_elements(std::vector<TmpFile>& temp_files) {
    return std::ranges::fold_left(count_elements<ElemT>(temp_files), 0, std::plus{});
}

template <typename ElemT>
void merge_and_write(std::vector<TmpFile>& temp_files, std::ostream& out) {

    std::vector<size_t> remaining = count_elements<ElemT>(temp_files);

    size_t written = 0;
    while (!temp_files.empty()) {
        std::uniform_int_distribution<size_t> dist(0, temp_files.size() - 1);
        const size_t                           idx = dist(rng::get_thread_local_rng());

        ElemT                           entry;
        temp_files[idx].ifstream().read(reinterpret_cast<char*>(&entry), sizeof(entry));
        out.write(reinterpret_cast<const char*>(&entry), sizeof(entry));

        --remaining[idx];
        ++written;

        if (written % 1000000 == 0) {
            std::cout << " positions written: " << written << std::endl;
        }
        if (remaining[idx] == 0) {
            temp_files.erase(temp_files.begin() + static_cast<long>(idx));
            remaining.erase(remaining.begin() + static_cast<long>(idx));
        }

        out.flush();
    }
}

#endif // CHEPP_DATA_LOADER_H
