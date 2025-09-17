//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_BINPACK_SFEN_INPUT_STREAM_H
#define CHEPP_BINPACK_SFEN_INPUT_STREAM_H

#include "steam_source.h"
#include "../binpack/nnue_training_data_stream.h"
#include "../utils/rng.h"
#include <functional>
#include <optional>


struct DataloaderSkipConfig {
    bool filtered;
    int  random_fen_skipping;
    bool wld_filtered;
    int  early_fen_skipping;
    int  simple_eval_skipping;
    int  param_index;
};

// this function was taken from the stockfish repo
inline std::function<bool(const binpack::TrainingDataEntry&)>
    make_skip_predicate(DataloaderSkipConfig config) {
    if (config.filtered || config.random_fen_skipping || config.wld_filtered
        || config.early_fen_skipping) {
        return [config, prob = static_cast<double>(config.random_fen_skipping) / (config.random_fen_skipping + 1)](
                   const binpack::TrainingDataEntry& e) {
            static constexpr int    VALUE_NONE                      = 32002;

            static constexpr double desired_piece_count_weights[33] = {
                1.000000, 1.121094, 1.234375, 1.339844, 1.437500, 1.527344, 1.609375,
                1.683594, 1.750000, 1.808594, 1.859375, 1.902344, 1.937500, 1.964844,
                1.984375, 1.996094, 2.000000, 1.996094, 1.984375, 1.964844, 1.937500,
                1.902344, 1.859375, 1.808594, 1.750000, 1.683594, 1.609375, 1.527344,
                1.437500, 1.339844, 1.234375, 1.121094, 1.000000};

            static constexpr double desired_piece_count_weights_total = []() {
                double tot = 0;
                for (auto w : desired_piece_count_weights)
                    tot += w;
                return tot;
            }();

            thread_local double       alpha                            = 1;
            thread_local double       piece_count_history_all[33]      = {0};
            thread_local double       piece_count_history_passed[33]   = {0};
            thread_local double       piece_count_history_all_total    = 0;
            thread_local double       piece_count_history_passed_total = 0;

            static constexpr double          max_skipping_rate                = 10.0;

            auto                             do_wld_skip                      = [&]() {
                std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() { return (e.isCapturingMove() || e.isInCheck()); };

            if (e.score == VALUE_NONE)
                return true;

            if (e.ply <= config.early_fen_skipping)
                return true;

            if (config.random_fen_skipping && do_skip())
                return true;

            if (config.filtered && do_filter())
                return true;

            if (config.wld_filtered && do_wld_skip())
                return true;

            if (config.simple_eval_skipping > 0
                && std::abs(e.pos.simple_eval()) < config.simple_eval_skipping)
                return true;

            const int pc = e.pos.piecesBB().count();
            piece_count_history_all[pc] += 1;
            piece_count_history_all_total += 1;

            if (static_cast<uint64_t>(piece_count_history_all_total) % 10000 == 0) {
                double pass = piece_count_history_all_total * desired_piece_count_weights_total;
                for (int i = 0; i < 33; ++i) {
                    if (desired_piece_count_weights[pc] > 0) {
                        double tmp =
                            piece_count_history_all_total * desired_piece_count_weights[pc]
                            / (desired_piece_count_weights_total * piece_count_history_all[pc]);
                        if (tmp < pass)
                            pass = tmp;
                    }
                }
                alpha = 1.0 / (pass * max_skipping_rate);
            }

            double tmp = alpha * piece_count_history_all_total * desired_piece_count_weights[pc]
                         / (desired_piece_count_weights_total * piece_count_history_all[pc]);
            tmp = std::min(1.0, tmp);
            if (std::bernoulli_distribution(1.0 - tmp)( rng::get_thread_local_rng()))
                return true;

            piece_count_history_passed[pc] += 1;
            piece_count_history_passed_total += 1;

            return false;
        };
    }

    return nullptr;
}

struct FilteredBinpackSfenInputStream : StreamSource<binpack::TrainingDataEntry> {
    training_data::BinpackSfenInputStream in;
    std::size_t remaining = 0;

    FilteredBinpackSfenInputStream(
        const std::string& path, const bool cyclic, const std::size_t n,
        std::function<bool(const training_data::TrainingDataEntry&)> skipPredicate
    ) : in(path, cyclic, std::move(skipPredicate)), remaining(n) {}

    std::optional<binpack::TrainingDataEntry> next() override {
        if (remaining-- == 0) return std::nullopt;
        return in.next();
    }
};

#endif // CHEPP_BINPACK_SFEN_INPUT_STREAM_H
