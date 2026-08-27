#ifndef WEATHER_CORE_H
#define WEATHER_CORE_H

#include <bits/stdc++.h>

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

    std::unordered_map<long long,long long> ivl;  // interval_id -> count

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
        ivl[r.ts / 60]++;

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
    for (int i = 0; i < src.S; ++i) {
        dst.st_count[i]   += src.st_count[i];
        dst.st_sumtemp[i] += src.st_sumtemp[i];
        dst.st_sumrain[i] += src.st_sumrain[i];
    }
    for (const auto& kv : src.ivl) dst.ivl[kv.first] += kv.second;

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
    add_kv("TOTAL_RAINFALL",      fmt6(s.sum_rain > -INFINITY ? s.sum_rain : 0.0));
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

    // Busiest 60-second interval: max count, ties -> smaller interval id.
    long long best_iv = 0, best_cnt = 0; bool first = true;
    for (const auto& kv : s.ivl) {
        if (first || kv.second > best_cnt || (kv.second == best_cnt && kv.first < best_iv)) {
            best_iv = kv.first; best_cnt = kv.second; first = false;
        }
    }
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
