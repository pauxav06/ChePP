//
// Created by paul on 7/30/25.
//

#ifndef TT_H
#define TT_H

#include "types.h"

#include "ChePP/engine/zobrist.h"
#include <bits/shared_ptr_base.h>
#include <optional>
#include <vector>

inline uint64_t floor_power_of_two(const uint64_t x)
{
    if (x == 0)
        return 0;
    return 1ULL << (63 - __builtin_clzll(x));
}

template <typename F, typename EntryT>
concept ReplacementPolicy = requires(F f, const EntryT& e1, const EntryT& e2) {
    { f(e1, e2) } -> std::convertible_to<bool>;
};

struct TT
{
    enum Bound : uint8_t
    {
        EXACT,
        LOWER,
        UPPER,
    };

    struct Entry
    {

        Entry() noexcept = default;
        Entry(const hash_t hash, const int depth, const int score, const Bound bound, const int generation,
              const Move move, int static_eval)
            : hash(hash), depth(depth), score(static_cast<int16_t>(score)), move(move), bound(bound), static_eval(static_eval),
              generation(generation)
        {
        }

        hash_t   hash{};
        uint16_t depth{};
        int16_t  score{};
        Move     move{};
        Bound    bound{};
        int16_t  static_eval{};
        uint8_t  generation{};
        uint8_t  repetitions{0};
    };

    void init(const size_t mb)
    {
        m_generation = 0;
        m_size = floor_power_of_two(mb * 1024 * 1024 / sizeof(Entry));
        m_table.resize(m_size);
        std::ranges::fill(m_table, Entry());
        std::cout << "Init tt with " << m_size << " entries" << std::endl;
    }

    void reset()
    {
        m_generation = 0;
        std::ranges::fill(m_table, Entry());
    }

    void prefetch(const hash_t hash) const noexcept
    {
        const size_t idx = index(hash);
        __builtin_prefetch(&m_table[idx], 0, 3);
    }

    [[nodiscard]] std::optional<Entry> probe(const hash_t hash) const
    {

        const Entry& cur = m_table[index(hash)];
        if (cur.hash != hash)
        {
            return std::nullopt;
        }
        return cur;
    }


    template <typename PolicyF>
    requires ReplacementPolicy<PolicyF, Entry>
    void store(PolicyF replacement_policy, const hash_t hash, const int depth, const int score, Bound bound, const Move move, int static_eval)
    {
        if (const auto entry = Entry(hash, depth, score, bound, m_generation, move, static_eval);
            replacement_policy(m_table[index(hash)], entry))
        {
            m_table[index(hash)] = entry;
        }
    }

    void new_generation() { m_generation++; }


    static int store_score(const int score, const int ply)
    {
        if (score >= MATE_IN_MAX_PLY)
            return score + ply;
        if (score <= MATED_IN_MAX_PLY)
            return score - ply;
        return score;
    };

    static int read_score(const int score, const int ply)
    {
        if (score >= MATE_IN_MAX_PLY)
            return score - ply;
        if (score <= MATED_IN_MAX_PLY)
            return score + ply;
        return score;
    };


  private:
    [[nodiscard]] size_t index(const hash_t hash) const { return hash & (m_size - 1); }

    int                m_generation{0};
    std::size_t        m_size{0};
    std::vector<Entry> m_table{};
};

#endif // TT_H
