#ifndef TT_H
#define TT_H

#include "core.h"
#include "format.h"

#include <bit>
#include <optional>
#include <vector>

#include <hwy/cache_control.h>

namespace chepp {
    inline uint64_t
    floor_power_of_two(const uint64_t x) {
        if (x == 0) return 0;
        return 1ULL << (63 - std::countl_zero(x));
    }

    template <typename F, typename EntryT>
    concept ReplacementPolicy = requires(F f, const EntryT& e1, const EntryT& e2) {
        { f(e1, e2) } -> std::convertible_to<bool>;
    };

    struct TT {
        enum Bound : uint8_t {
            EXACT,
            LOWER,
            UPPER,
        };

        struct Entry {

            Entry() noexcept = default;
            Entry(const zobrist::Hash hash,
                  const uint16_t      depth,
                  const int16_t       score,
                  const Bound         bound,
                  const uint8_t       generation,
                  const Move          move,
                  const int16_t       static_eval)
                : hash(hash), depth(depth), score(score), move(move), bound(bound), static_eval(static_eval),
                  generation(generation) {
            }

            zobrist::Hash hash{0};
            uint16_t      depth{0};
            int16_t       score{0};
            Move          move{Move::none()};
            Bound         bound{Bound::EXACT};
            int16_t       static_eval{0};
            uint8_t       generation{0};
            uint8_t       repetitions{0};
        };

        void
        init(const size_t mb) {
            m_generation = 1;
            m_size       = floor_power_of_two(mb * 1024 * 1024 / sizeof(Entry));
            m_table.resize(m_size);
            std::ranges::fill(m_table, Entry());
            fmt::println(stdout, "Init tt with {} entries ", m_size);
        }

        void
        reset() {
            m_generation = 1;
            ranges::fill(m_table, Entry());
        }

        void
        prefetch(const zobrist::Hash hash) const noexcept {
            const size_t idx = index(hash);
            hwy::Prefetch(&m_table[idx]);
        }

        [[nodiscard]] std::optional<Entry>
        probe(const zobrist::Hash hash) const {

            const Entry& cur = m_table[index(hash)];
            if (cur.hash != hash) {
                return std::nullopt;
            }
            return cur;
        }

        void
        store(const zobrist::Hash hash,
              const uint16_t      depth,
              const int16_t       score,
              Bound               bound,
              const Move          move,
              int16_t             static_eval) {
            const auto candidate = Entry(hash, depth, score, bound, m_generation, move, static_eval);
            auto&      old       = m_table[index(hash)];

            bool replace = (old.hash != hash) || (old.generation != candidate.generation || old.depth <= candidate.depth);
            if (!replace) return;
            if (candidate.move || old.hash != candidate.hash) old.move = move;
            if (candidate.bound == TT::EXACT || candidate.hash != old.hash || candidate.depth + 4 > old.depth) {
                old = candidate;
            }
        }

        void
        new_generation() {
            m_generation++;
        }

        static int
        store_score(const int score, const int ply) {
            if (score >= MATE_IN_MAX_PLY) return score + ply;
            if (score <= MATED_IN_MAX_PLY) return score - ply;
            return score;
        };

        static int
        read_score(const int score, const int ply) {
            if (score >= MATE_IN_MAX_PLY) return score - ply;
            if (score <= MATED_IN_MAX_PLY) return score + ply;
            return score;
        };

      private:
        [[nodiscard]] size_t
        index(const zobrist::Hash hash) const {
            return hash & (m_size - 1);
        }

        uint8_t            m_generation{0};
        std::size_t        m_size{0};
        std::vector<Entry> m_table{};
    };
} // namespace chepp

#endif // TT_H
