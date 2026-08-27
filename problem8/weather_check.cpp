#include "weather_core.h"
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s input.txt\n", argv[0]); return 1; }
    ifstream fin(argv[1]);
    long long N; int K, S; fin >> N >> K >> S;
    vector<Rec> recs(N);
    for (long long i = 0; i < N; ++i)
        fin >> recs[i].ts >> recs[i].sid >> recs[i].temp >> recs[i].hum
            >> recs[i].pres >> recs[i].rain >> recs[i].wind;

    // single pass
    Stats one; one.init(S);
    for (auto& r : recs) one.add(r);
    string ref = format_results(one, K);

    int Ps[4] = {1, 2, 4, 8};
    bool ok = true;
    for (int P : Ps) {
        // split like MPI Scatterv (first N%P chunks get one extra record)
        Stats global; global.init(S);
        long long base = N / P, rem = N % P, off = 0;
        for (int p = 0; p < P; ++p) {
            long long cnt = base + (p < rem ? 1 : 0);
            Stats local; local.init(S);
            for (long long i = 0; i < cnt; ++i) local.add(recs[off + i]);
            off += cnt;
            merge_into(global, local);
        }
        string got = format_results(global, K);
        if (got != ref) { printf("MISMATCH at P=%d\n", P); ok = false; }
    }
    if (ok) printf("OK  N=%lld K=%d S=%d  split-merge matches sequential for P=1,2,4,8\n", N, K, S);
    return ok ? 0 : 1;
}
