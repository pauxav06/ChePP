#ifndef PRNG_H
#define PRNG_H
#include <cstdint>
#include <string_view>

constexpr uint64_t fnv1a64(const std::string_view str, uint64_t hash = 14695981039346656037ULL) {
    for (const char c : str) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

constexpr uint64_t combine(const uint64_t a, const uint64_t b) {
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 12) + (a >> 4));
}

constexpr uint64_t make_seed_(const std::string_view file, const std::string_view func, const uint64_t line) {
    uint64_t h = fnv1a64(file);
    h = combine(h, fnv1a64(func));
    h = combine(h, line);
    return h;
}

// XOR shift algo, need to check exactly what it does, just use for now
// http://vigna.di.unimi.it/ftp/papers/xorshift.pdf
struct PRNG {
    uint64_t s;

    constexpr explicit PRNG(const uint64_t seed) : s(seed) {}

    constexpr PRNG next(uint64_t& out) const {
        uint64_t x = s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        out = x * 2685821657736338717ULL;
        return PRNG(x);
    }
    constexpr auto next_rand(uint64_t& out) const {
        uint64_t r1, r2, r3;
        PRNG g1 = next(r1);
        PRNG g2 = g1.next(r2);
        PRNG g3 = g2.next(r3);
        out = r1 & r2 & r3;
        return g3;
    }
};


#endif
