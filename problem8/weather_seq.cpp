#include <chrono>
#include "weather_core.h"
using namespace std;

// Sequential reference. Parsing and analytics are timed SEPARATELY and on
// purpose: the MPI program's timed region excludes rank 0's file parse, so the
// only like-for-like baseline for it is this program's analytics phase. Timing
// the parse in one and not the other would manufacture a speed-up that is really
// just "text parsing is slow".
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s input.txt [output.txt]\n", argv[0]); return 1; }
    ifstream fin(argv[1]);
    if (!fin) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 2; }

    long long N; int K, S;
    fin >> N >> K >> S;

    // --- parse phase (not part of the comparable measurement) ---
    auto p0 = chrono::high_resolution_clock::now();
    vector<Rec> recs((size_t)N);
    for (long long i = 0; i < N; ++i)
        fin >> recs[i].ts >> recs[i].sid >> recs[i].temp >> recs[i].hum
            >> recs[i].pres >> recs[i].rain >> recs[i].wind;
    if (!fin) { fprintf(stderr, "Malformed input %s: expected %lld records\n", argv[1], N); return 2; }
    auto p1 = chrono::high_resolution_clock::now();

    // --- analytics phase (the baseline the MPI timings are compared against) ---
    Stats s; s.init(S);
    for (const auto& r : recs) s.add(r);
    auto p2 = chrono::high_resolution_clock::now();

    double parse     = chrono::duration<double>(p1 - p0).count();
    double analytics = chrono::duration<double>(p2 - p1).count();
    fprintf(stderr, "SEQ  N=%lld K=%d S=%d  time=%.6f s  parse=%.6f analytics=%.6f\n",
            N, K, S, analytics, parse, analytics);

    string out = format_results(s, K);
    if (argc >= 3) { ofstream f(argv[2]); f << out; }
    else           { fputs(out.c_str(), stdout); }
    return 0;
}
