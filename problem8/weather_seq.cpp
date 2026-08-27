#include "weather_core.h"
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s input.txt [output.txt]\n", argv[0]); return 1; }
    ifstream fin(argv[1]);
    if (!fin) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 2; }

    long long N; int K, S;
    fin >> N >> K >> S;

    Stats s; s.init(S);
    auto t0 = chrono::high_resolution_clock::now();
    Rec r;
    for (long long i = 0; i < N; ++i) {
        fin >> r.ts >> r.sid >> r.temp >> r.hum >> r.pres >> r.rain >> r.wind;
        s.add(r);
    }
    auto t1 = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "SEQ  N=%lld K=%d S=%d  time=%.6f s\n", N, K, S, secs);

    string out = format_results(s, K);
    if (argc >= 3) { ofstream f(argv[2]); f << out; }
    else           { fputs(out.c_str(), stdout); }
    return 0;
}
