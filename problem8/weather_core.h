#ifndef WEATHER_CORE_H
#define WEATHER_CORE_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// One weather measurement. POD so it can be sent as raw bytes over MPI.
struct Rec {
    long long ts;    // timestamp (integer)
    int       sid;   // station_id (integer, 0..S-1)
    double    temp, hum, pres, rain, wind;
};

// A single extreme (hottest/coldest) measurement, with tie-break fields.
struct Meas {
    double    temp = 0.0;
    long long ts   = 0;
    int       sid  = 0;
    bool      valid = false;
};

// Tie-break: hottest = highest temp; ties -> smaller timestamp, then smaller sid.
inline bool is_hotter(const Meas& a, const Meas& b) {
    if (!b.valid) return true;
    if (!a.valid) return false;
    if (a.temp != b.temp) return a.temp > b.temp;
    if (a.ts   != b.ts)   return a.ts   < b.ts;
    return a.sid < b.sid;
}
// Coldest = lowest temp; ties -> smaller timestamp, then smaller sid.
inline bool is_colder(const Meas& a, const Meas& b) {
    if (!b.valid) return true;
    if (!a.valid) return false;
    if (a.temp != b.temp) return a.temp < b.temp;
    if (a.ts   != b.ts)   return a.ts   < b.ts;
    return a.sid < b.sid;
}

// Safety valve for the dense interval histogram (see IntervalHist below).
// 200M buckets = 1.6 GB; anything beyond that is a pathological timestamp range.
static const long long IVL_MAX_BUCKETS = 200000000LL;

// Dense histogram of 60-second interval ids.
//
// Earlier revisions used unordered_map<interval_id,count>. That is O(1) in
// theory but disastrous here: the number of DISTINCT intervals grows with N
// (~N/4 for the supplied generator), so every rank ended up hashing one entry
// per record and then shipping the whole map to rank 0 to be merged serially.
// A dense array indexed by (iv - base) is ~30x faster to build, and — more
// importantly — it is a plain contiguous buffer, so combining partial results
// becomes a single MPI_Reduce(MPI_SUM) instead of a gather-and-hash-merge.
struct IntervalHist {
    long long base = 0;               // interval id of element 0
    std::vector<long long> c;         // c[i] = count for interval (base + i)

    bool empty() const { return c.empty(); }
    long long lo() const { return base; }                        // inclusive
    long long hi() const { return base + (long long)c.size(); }  // exclusive

    // Grow geometrically so a scattered arrival order stays amortised O(1)/record.
    void ensure(long long iv) {
        if (c.empty()) { base = iv; c.assign(1, 0); return; }
        if (iv < base) {
            long long need = base - iv;
            long long grow = std::max(need, (long long)c.size());
            if ((long long)c.size() + grow > IVL_MAX_BUCKETS) {
                std::fprintf(stderr, "Interval range too large (>%lld buckets).\n", IVL_MAX_BUCKETS);
                std::exit(3);
            }
            c.insert(c.begin(), (size_t)grow, 0);
            base -= grow;
        }
        if (iv - base >= (long long)c.size()) {
            long long need = iv - base + 1;
            long long grow = std::max(need, (long long)c.size() * 2);
            if (grow > IVL_MAX_BUCKETS) {
                std::fprintf(stderr, "Interval range too large (>%lld buckets).\n", IVL_MAX_BUCKETS);
                std::exit(3);
            }
            c.resize((size_t)grow, 0);
        }
    }
    void bump(long long iv) { ensure(iv); c[(size_t)(iv - base)]++; }

    // Re-index onto [nbase, nbase+nsize) so two histograms can be added
    // elementwise — and so every MPI rank can agree on one common layout.
    void rebase(long long nbase, long long nsize) {
        std::vector<long long> n((size_t)nsize, 0);
        for (size_t i = 0; i < c.size(); ++i) {
            long long iv = base + (long long)i;
            if (iv >= nbase && iv < nbase + nsize) n[(size_t)(iv - nbase)] += c[i];
        }
        base = nbase; c.swap(n);
    }

    void merge(const IntervalHist& o) {
        if (o.empty()) return;
        if (empty()) { *this = o; return; }
        long long nb = std::min(lo(), o.lo()), ne = std::max(hi(), o.hi());
        if (nb != lo() || ne != hi()) rebase(nb, ne - nb);
        for (size_t i = 0; i < o.c.size(); ++i) c[(size_t)(o.base + (long long)i - base)] += o.c[i];
    }

    // Busiest interval: max count; ties -> smaller interval id. Scanning
    // ascending with a strict '>' keeps the smallest id on a tie.
    void busiest(long long& iv, long long& cnt) const {
        iv = 0; cnt = 0;
        for (size_t i = 0; i < c.size(); ++i)
            if (c[i] > cnt) { cnt = c[i]; iv = base + (long long)i; }
    }
};

struct Stats {
    long long total = 0;

    double sum_temp = 0, min_temp =  INFINITY, max_temp = -INFINITY;
    double sum_hum  = 0, min_hum  =  INFINITY, max_hum  = -INFINITY;
    double sum_pres = 0, min_pres =  INFINITY, max_pres = -INFINITY;
    double sum_rain = 0,                        max_rain = -INFINITY;
    double sum_wind = 0,                        max_wind = -INFINITY;

    long long extreme = 0;               // temp >= 40.0 OR temp <= 0.0

    Meas hot, cold;

    int S = 0;                           // number of stations
    std::vector<long long> st_count;     // per-station measurement count
    std::vector<double>    st_sumtemp;   // per-station temperature sum
    std::vector<double>    st_sumrain;   // per-station rainfall sum

    IntervalHist ivl;                    // 60-second interval histogram

    void init(int S_) {
        S = S_;
        st_count.assign(S, 0);
        st_sumtemp.assign(S, 0.0);
        st_sumrain.assign(S, 0.0);
    }

    void add(const Rec& r) {
        total++;
        sum_temp += r.temp; min_temp = std::min(min_temp, r.temp); max_temp = std::max(max_temp, r.temp);
        sum_hum  += r.hum;  min_hum  = std::min(min_hum,  r.hum);  max_hum  = std::max(max_hum,  r.hum);
        sum_pres += r.pres; min_pres = std::min(min_pres, r.pres); max_pres = std::max(max_pres, r.pres);
        sum_rain += r.rain;                                        max_rain = std::max(max_rain, r.rain);
        sum_wind += r.wind;                                        max_wind = std::max(max_wind, r.wind);
        if (r.temp >= 40.0 || r.temp <= 0.0) extreme++;

        if (r.sid >= 0 && r.sid < S) {
            st_count[r.sid]++;
            st_sumtemp[r.sid] += r.temp;
            st_sumrain[r.sid] += r.rain;
        }
        ivl.bump(r.ts / 60);

        Meas m{r.temp, r.ts, r.sid, true};
        if (is_hotter(m, hot)) hot = m;
        if (is_colder(m, cold)) cold = m;
    }
};

// Combine src into dst (dst += src). Order-independent for the discrete parts.
inline void merge_into(Stats& dst, const Stats& src) {
    dst.total    += src.total;
    dst.sum_temp += src.sum_temp; dst.min_temp = std::min(dst.min_temp, src.min_temp); dst.max_temp = std::max(dst.max_temp, src.max_temp);
    dst.sum_hum  += src.sum_hum;  dst.min_hum  = std::min(dst.min_hum,  src.min_hum);  dst.max_hum  = std::max(dst.max_hum,  src.max_hum);
    dst.sum_pres += src.sum_pres; dst.min_pres = std::min(dst.min_pres, src.min_pres); dst.max_pres = std::max(dst.max_pres, src.max_pres);
    dst.sum_rain += src.sum_rain;                                                      dst.max_rain = std::max(dst.max_rain, src.max_rain);
    dst.sum_wind += src.sum_wind;                                                      dst.max_wind = std::max(dst.max_wind, src.max_wind);
    dst.extreme  += src.extreme;

    if (dst.S == 0 && src.S > 0) dst.init(src.S);
    for (int i = 0; i < src.S && i < dst.S; ++i) {
        dst.st_count[i]   += src.st_count[i];
        dst.st_sumtemp[i] += src.st_sumtemp[i];
        dst.st_sumrain[i] += src.st_sumrain[i];
    }
    dst.ivl.merge(src.ivl);

    if (is_hotter(src.hot, dst.hot)) dst.hot = src.hot;
    if (is_colder(src.cold, dst.cold)) dst.cold = src.cold;
}

inline std::string fmt6(double x) { char b[64]; std::snprintf(b, sizeof b, "%.6f", x); return b; }

// Produce the exact required output text for a fully-merged Stats.
inline std::string format_results(const Stats& s, int K) {
    std::string o;
    auto add_kv = [&](const char* k, const std::string& v){ o += k; o += ' '; o += v; o += '\n'; };
    bool any = s.total > 0;

    add_kv("TOTAL_MEASUREMENTS", std::to_string(s.total));
    add_kv("AVERAGE_TEMPERATURE", fmt6(any ? s.sum_temp / s.total : 0.0));
    add_kv("MIN_TEMPERATURE",     fmt6(any ? s.min_temp : 0.0));
    add_kv("MAX_TEMPERATURE",     fmt6(any ? s.max_temp : 0.0));
    add_kv("AVERAGE_HUMIDITY",    fmt6(any ? s.sum_hum / s.total : 0.0));
    add_kv("MIN_HUMIDITY",        fmt6(any ? s.min_hum : 0.0));
    add_kv("MAX_HUMIDITY",        fmt6(any ? s.max_hum : 0.0));
    add_kv("AVERAGE_PRESSURE",    fmt6(any ? s.sum_pres / s.total : 0.0));
    add_kv("MIN_PRESSURE",        fmt6(any ? s.min_pres : 0.0));
    add_kv("MAX_PRESSURE",        fmt6(any ? s.max_pres : 0.0));
    add_kv("TOTAL_RAINFALL",      fmt6(s.sum_rain));
    add_kv("MAX_RAINFALL",        fmt6(any ? s.max_rain : 0.0));
    add_kv("AVERAGE_WIND_SPEED",  fmt6(any ? s.sum_wind / s.total : 0.0));
    add_kv("MAX_WIND_SPEED",      fmt6(any ? s.max_wind : 0.0));
    add_kv("EXTREME_TEMPERATURE_EVENTS", std::to_string(s.extreme));

    o += "HOTTEST_MEASUREMENT " + fmt6(s.hot.valid ? s.hot.temp : 0.0) + " " +
         std::to_string(s.hot.valid ? s.hot.sid : 0) + " " +
         std::to_string(s.hot.valid ? s.hot.ts : 0) + "\n";
    o += "COLDEST_MEASUREMENT " + fmt6(s.cold.valid ? s.cold.temp : 0.0) + " " +
         std::to_string(s.cold.valid ? s.cold.sid : 0) + " " +
         std::to_string(s.cold.valid ? s.cold.ts : 0) + "\n";

    long long best_iv, best_cnt;
    s.ivl.busiest(best_iv, best_cnt);
    o += "BUSIEST_INTERVAL " + std::to_string(best_iv) + " " + std::to_string(best_cnt) + "\n";

    // Top-K stations: count desc, then station id asc. Only stations with count>0.
    o += "TOP_STATIONS\n";
    std::vector<int> ids;
    for (int i = 0; i < s.S; ++i) if (s.st_count[i] > 0) ids.push_back(i);
    std::sort(ids.begin(), ids.end(), [&](int a, int b){
        if (s.st_count[a] != s.st_count[b]) return s.st_count[a] > s.st_count[b];
        return a < b;
    });
    int lim = std::min((int)ids.size(), K);
    for (int i = 0; i < lim; ++i) {
        int id = ids[i];
        double avgt = s.st_sumtemp[id] / s.st_count[id];
        o += std::to_string(id) + " " + std::to_string(s.st_count[id]) + " " +
             fmt6(avgt) + " " + fmt6(s.st_sumrain[id]) + "\n";
    }
    return o;
}

#endif
