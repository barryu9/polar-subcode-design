#ifdef _WIN32
#define PAPER_EXPORT __declspec(dllexport)
#else
#define PAPER_EXPORT __attribute__((visibility("default")))
#endif
// Exact distinct-coordinate syndrome counts through radius eight.
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdexcept>

struct Key {
    uint64_t lo, hi;
    bool operator==(const Key& other) const { return lo == other.lo && hi == other.hi; }
};
static Key operator^(Key a, Key b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
struct Hash {
    size_t operator()(Key a) const {
        auto mix = [](uint64_t x) {
            x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27; x *= 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        return mix(a.lo) ^ mix(a.hi + 0x9e3779b97f4a7c15ULL);
    }
};
using Table = std::unordered_map<Key, uint64_t, Hash>;
static uint64_t choose(int n, int k) {
    if (k < 0 || n < k) return 0;
    uint64_t v = 1;
    for (int i = 1; i <= k; ++i) v = v * (n-i+1) / i;
    return v;
}

extern "C" PAPER_EXPORT int relative_counts(
    int n, const uint64_t* low, const uint64_t* high,
    uint64_t target_low, uint64_t target_high, int radius, uint64_t* out) {
    try {
        if (n < 0 || n > 1024 || radius < 0 || radius > 8 || (n > 128 && radius > 4)) return 1;
        std::vector<Key> cols;
        for (int i = 0; i < n; ++i) cols.push_back({low[i], high[i]});
        const Key target{target_low, target_high};
        std::vector<Table> tables(5);
        tables[0][{0, 0}] = 1;
        const int depth = (radius+1)/2;
        for (int t = 1; t <= depth; ++t)
            tables[t].reserve(std::min<uint64_t>(choose(n, t), 4000000));
        for (int i = 0; i < n && depth >= 1; ++i) {
            ++tables[1][cols[i]];
            for (int j = i+1; j < n && depth >= 2; ++j) {
                const Key a = cols[i] ^ cols[j]; ++tables[2][a];
                for (int k = j+1; k < n && depth >= 3; ++k) {
                    const Key b = a ^ cols[k]; ++tables[3][b];
                    for (int l = k+1; l < n && depth >= 4; ++l)
                        ++tables[4][b ^ cols[l]];
                }
            }
        }
        for (int w = 0; w <= radius; ++w) {
            if (w <= depth) {
                auto hit = tables[w].find(target);
                out[w] = hit == tables[w].end() ? 0 : hit->second;
                continue;
            }
            const int a = w/2, b = w-a;
            __int128 sum = 0;
            for (const auto& item : tables[a]) {
                auto hit = tables[b].find(item.first ^ target);
                if (hit != tables[b].end()) sum += (__int128)item.second * hit->second;
            }
            // Intersecting halves cancel shared coordinates in the XOR syndrome.
            for (int j = 1; j <= a; ++j) {
                const int t = w-2*j;
                sum -= (__int128)choose(t, a-j) * choose(n-t, j) * out[t];
            }
            const auto partitions = choose(w, a);
            if (sum < 0 || sum % partitions) return 2;
            out[w] = (uint64_t)(sum / partitions);
            if (sum / partitions > UINT64_MAX) return 3;
        }
        return 0;
    } catch (...) { return 4; }
}
