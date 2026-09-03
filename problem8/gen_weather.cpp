// gen_weather.cpp — reproducible dataset generator for Q8.
//   g++ -O2 -std=c++17 -o gen_weather gen_weather.cpp
//   ./gen_weather N K S seed [outfile]     # outfile omitted -> stdout
//
// Produces:
//   line 1 : N K S
//   next N : timestamp station_id temperature humidity pressure rainfall wind_speed
//
// Value ranges (documented for reproducibility), all floats to 1 decimal place:
//   timestamp   : integer, uniform in [0, 60*N/4]  (a few records per 60s bucket)
//   station_id  : integer, uniform in [0, S-1]
//   temperature : [-10.0, 45.0]   (spans the <=0 and >=40 extreme bands)
//   humidity    : [0.0, 100.0]
//   pressure    : [950.0, 1050.0]
//   rainfall    : [0.0, 50.0]
//   wind_speed  : [0.0, 150.0]
// Deterministic: same (N,K,S,seed) always yields the same file (mt19937).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
using namespace std;

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "Usage: %s N K S seed [outfile]\n", argv[0]); return 1; }
    long long N = atoll(argv[1]);
    int K = atoi(argv[2]);
    int S = atoi(argv[3]);
    unsigned seed = (unsigned)strtoul(argv[4], nullptr, 10);

    FILE* out = stdout;
    if (argc >= 6) { out = fopen(argv[5], "w"); if (!out) { fprintf(stderr, "Cannot open %s\n", argv[5]); return 2; } }

    mt19937 rng(seed);
    long long tsMax = max<long long>(60, 60LL * N / 4);
    uniform_int_distribution<long long> dTs(0, tsMax);
    uniform_int_distribution<int>       dSid(0, S - 1);
    auto d1 = [](int lo, int hi){ return uniform_int_distribution<int>(lo, hi); };
    auto dTemp = d1(-100, 450);   // /10 -> [-10.0, 45.0]
    auto dHum  = d1(0, 1000);     // /10 -> [0.0, 100.0]
    auto dPres = d1(9500, 10500); // /10 -> [950.0, 1050.0]
    auto dRain = d1(0, 500);      // /10 -> [0.0, 50.0]
    auto dWind = d1(0, 1500);     // /10 -> [0.0, 150.0]

    fprintf(out, "%lld %d %d\n", N, K, S);
    for (long long i = 0; i < N; ++i) {
        long long ts = dTs(rng);
        int sid = dSid(rng);
        fprintf(out, "%lld %d %.1f %.1f %.1f %.1f %.1f\n",
                ts, sid,
                dTemp(rng) / 10.0, dHum(rng) / 10.0, dPres(rng) / 10.0,
                dRain(rng) / 10.0, dWind(rng) / 10.0);
    }
    if (out != stdout) fclose(out);
    return 0;
}
