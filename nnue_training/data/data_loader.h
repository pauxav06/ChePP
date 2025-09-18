//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_DATA_LOADER_H
#define CHEPP_DATA_LOADER_H

#include "nnue_training_data_formats.h"
#include "stream_view.h"
#include "utils/utils.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <ranges>
#include <vector>

template <typename ElemT>
TmpFile process_chunk(std::vector<ElemT> chunk)
{
    TmpFile tmp;
    auto&   ofs = tmp.stream();

    std::ranges::shuffle(chunk, rng::get_thread_local_rng());
    std::ranges::for_each(chunk, [&ofs](auto i) { ofs.write(reinterpret_cast<char*>(&i), sizeof(i)); });
    ofs.flush();

    return tmp;
}

// this function reads from the stream, performs conversion, chunks data.
// data in one chunk is shuffled and written to a temporary file
template <typename InputT, typename OutputT>
auto convert_and_shuffle_chunks(StreamView<InputT>&&                  sv,
                                const std::function<OutputT(const InputT&)>& converter,
                                const double val_split = 0.1)
{
    using chunk_t = std::vector<OutputT>;

    static constexpr size_t chunk_size = 4096*256; // about 40mb per file

    std::vector<std::future<TmpFile>> futures{};
    auto view_chunk = sv | std::views::transform(converter) | std::views::chunk(chunk_size);
    auto to_future  = [&](auto&& view)
    {
        chunk_t chunk;
        chunk.reserve(chunk_size);
        std::ranges::copy(view, std::back_inserter(chunk));
        return std::async(std::launch::async, process_chunk<OutputT>, std::move(chunk));
    };
    std::ranges::transform(view_chunk, std::back_inserter(futures), to_future);

    std::vector<TmpFile> train_files, val_files;
    for (auto& fut : futures)
    {
        (std::bernoulli_distribution(1.0 - val_split)(rng::get_thread_local_rng()) ? train_files : val_files)
            .push_back(fut.get());
    }

    for (auto& file : train_files) { file.stream().seekg(0, std::ios::beg);}
    for (auto& file : val_files) { file.stream().seekg(0, std::ios::beg);}

    return std::pair{std::move(train_files), std::move(val_files)};
}

template <typename ElemT>
size_t count_elements(TmpFile& file)
{
    auto&                        stream = file.stream();
    const std::istream::pos_type pos    = stream.tellg();
    stream.seekg(0, std::ios::end);
    const size_t c = static_cast<size_t>(stream.tellg()) / sizeof(ElemT);
    stream.seekg(pos);
    return c;
}

template <typename ElemT>
std::vector<size_t> count_elements(std::vector<TmpFile>& temp_files)
{
    auto counts = temp_files | std::views::transform(count_elements<ElemT>);
    return std::vector(counts.begin(), counts.end());
}

template <typename ElemT>
size_t count_all_elements(std::vector<TmpFile>& temp_files)
{
    return std::ranges::fold_left(count_elements<ElemT>(temp_files), size_t{0}, std::plus<>{});
}


template <typename ElemT>
void merge_and_write(std::vector<TmpFile>& temp_files,
                     std::vector<std::ofstream>& out_streams,
                     size_t buffer_size = 65536)
{
    struct FileEntry {
        TmpFile* file;
        std::mutex mtx{}; //make sure the tmp file size is small enough so we do not get contention on mutex
        size_t remaining;

        FileEntry(TmpFile* f, const size_t rem) : file(f), remaining(rem)
        {}
    };

    std::vector<std::unique_ptr<FileEntry>> files;
    files.reserve(temp_files.size());
    for (auto& tf : temp_files) {
        files.push_back(std::make_unique<FileEntry>(&tf, count_elements<ElemT>(tf)));
    }

    std::atomic<int> done_count{0};

    const size_t n_threads = out_streams.size();
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_threads; ++i)
    {
        threads.emplace_back(
            [&, i]()
            {
                auto& out = out_streams.at(i);
                std::vector<ElemT> local_buffer;
                local_buffer.reserve(buffer_size);

                std::mt19937_64& rng = rng::get_thread_local_rng();
                std::uniform_int_distribution<size_t> dist(0, files.size() - 1);

                while (true)
                {
                    local_buffer.clear();
                    bool all_done = false;

                    while (local_buffer.size() < buffer_size)
                    {
                        all_done = true;
                        for (auto& f : files)
                            if (f->remaining > 0) { all_done = false; break; }

                        if (all_done) break;

                        FileEntry* fe = nullptr;
                        for (int tries = 0; tries < 10; ++tries) {
                            auto& candidate = *files[dist(rng)];
                            if (candidate.remaining > 0) {
                                fe = &candidate;
                                break;
                            }
                        }
                        if (!fe) continue;

                        ElemT entry;
                        {
                            std::unique_lock<std::mutex> lock(fe->mtx);
                            if (fe->remaining > 0) {
                                fe->file->stream().read(reinterpret_cast<char*>(&entry), sizeof(entry));
                                --fe->remaining;
                                local_buffer.push_back(entry);
                            }
                        }
                    }

                    if (!local_buffer.empty()) {
                        out.write(reinterpret_cast<const char*>(local_buffer.data()), local_buffer.size() * sizeof(ElemT));
                    }

                    if (all_done) break;
                }
                out.flush();
            });
    }

    for (auto& t : threads)
        t.join();

    temp_files.clear();
}


template <typename InputT, typename HeaderT, typename OutputT, typename Source>
requires (std::is_base_of_v<StreamSource<InputT>, Source>)
void binpack_convert(
    const std::vector<std::string>& input_files,
    const std::vector<std::string>& train_outputs,
    const std::vector<std::string>& val_outputs,
    const std::function<Source(const std::string&)>& stream_factory,
    const std::function<HeaderT(std::size_t)>& header_factory,
    const std::function<OutputT(const InputT&)>& converter,
    const float val_split)
{
    std::mutex           tmp_mutex;
    std::vector<TmpFile> shared_train_tmp, shared_val_tmp;

    std::cout << "reading input files" << std::endl;
    std::vector<std::future<void>> futures;
    for (auto& file : input_files)
    {
        futures.push_back(std::async(
            std::launch::async,
            [&](const std::string& f)
            {
                auto sv = stream_factory(f);
                auto [train_files, val_files] = convert_and_shuffle_chunks<InputT, OutputT>(StreamView<InputT>(sv), converter, val_split);

                std::scoped_lock lock(tmp_mutex);
                shared_train_tmp.insert(shared_train_tmp.end(), std::make_move_iterator(train_files.begin()),
                                        std::make_move_iterator(train_files.end()));
                shared_val_tmp.insert(shared_val_tmp.end(), std::make_move_iterator(val_files.begin()),
                                      std::make_move_iterator(val_files.end()));
            },
            file));
    }
    for (auto& fut : futures)
        fut.get();

    std::cout << "writing output files" << std::endl;
    auto to_stream = [] (const std::string& f) {
        return std::ofstream(f, std::ios::binary | std::ios::trunc);
    };

    auto train_out = train_outputs | std::views::transform(to_stream);
    auto val_out   = val_outputs   | std::views::transform(to_stream);
    std::vector<std::ofstream> train_outs{train_out.begin(), train_out.end()};
    std::vector<std::ofstream> val_outs{val_out.begin(), val_out.end()};

    auto write_header_with_n = [&header_factory] (const size_t n, std::ofstream& out) {
        out.seekp(0, std::ios::beg);
        HeaderT header = header_factory(n);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.flush();
    };
    for (auto& out : train_outs) write_header_with_n(0, out);
    for (auto& out : val_outs)   write_header_with_n(0, out);

    merge_and_write<OutputT>(shared_train_tmp, train_outs);
    merge_and_write<OutputT>(shared_val_tmp,   val_outs);

    for (auto& out : train_outs) out.close();
    for (auto& out : val_outs)   out.close();

    auto fix_header = [&](const std::string& fname) {
        std::fstream f(fname, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(0, std::ios::end);
        const auto end = f.tellg();
        const size_t n = (static_cast<size_t>(end) - sizeof(HeaderT)) / sizeof(OutputT);
        f.seekp(0, std::ios::beg);
        HeaderT header = header_factory(n);
        f.write(reinterpret_cast<const char*>(&header), sizeof(header));
        f.close();
    };
    for (auto& f : train_outputs) fix_header(f);
    for (auto& f : val_outputs)   fix_header(f);

}

#endif // CHEPP_DATA_LOADER_H
