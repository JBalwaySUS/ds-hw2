#include <mpi.h>
#include "weather_core.h"
using namespace std;

// ---- serialise / deserialise a Stats to/from a byte buffer ----
static void put(vector<char>& b, const void* p, size_t n) {
    const char* c = (const char*)p; b.insert(b.end(), c, c + n);
}
template <class T> static void putv(vector<char>& b, const T& x) { put(b, &x, sizeof(T)); }

static vector<char> serialize(const Stats& s) {
    vector<char> b;
    putv(b, s.total);
    putv(b, s.sum_temp); putv(b, s.min_temp); putv(b, s.max_temp);
    putv(b, s.sum_hum);  putv(b, s.min_hum);  putv(b, s.max_hum);
    putv(b, s.sum_pres); putv(b, s.min_pres); putv(b, s.max_pres);
    putv(b, s.sum_rain); putv(b, s.max_rain);
    putv(b, s.sum_wind); putv(b, s.max_wind);
    putv(b, s.extreme);
    putv(b, s.hot); putv(b, s.cold);
    putv(b, s.S);
    put(b, s.st_count.data(),   sizeof(long long) * s.S);
    put(b, s.st_sumtemp.data(), sizeof(double)    * s.S);
    put(b, s.st_sumrain.data(), sizeof(double)    * s.S);
    long long nIvl = (long long)s.ivl.size();
    putv(b, nIvl);
    for (const auto& kv : s.ivl) { putv(b, kv.first); putv(b, kv.second); }
    return b;
}

static const char* get(const char* p, void* dst, size_t n) { memcpy(dst, p, n); return p + n; }
template <class T> static const char* getv(const char* p, T& x) { return get(p, &x, sizeof(T)); }

static Stats deserialize(const char* p) {
    Stats s;
    p = getv(p, s.total);
    p = getv(p, s.sum_temp); p = getv(p, s.min_temp); p = getv(p, s.max_temp);
    p = getv(p, s.sum_hum);  p = getv(p, s.min_hum);  p = getv(p, s.max_hum);
    p = getv(p, s.sum_pres); p = getv(p, s.min_pres); p = getv(p, s.max_pres);
    p = getv(p, s.sum_rain); p = getv(p, s.max_rain);
    p = getv(p, s.sum_wind); p = getv(p, s.max_wind);
    p = getv(p, s.extreme);
    p = getv(p, s.hot); p = getv(p, s.cold);
    p = getv(p, s.S);
    s.st_count.resize(s.S); s.st_sumtemp.resize(s.S); s.st_sumrain.resize(s.S);
    p = get(p, s.st_count.data(),   sizeof(long long) * s.S);
    p = get(p, s.st_sumtemp.data(), sizeof(double)    * s.S);
    p = get(p, s.st_sumrain.data(), sizeof(double)    * s.S);
    long long nIvl; p = getv(p, nIvl);
    for (long long i = 0; i < nIvl; ++i) { long long k, v; p = getv(p, k); p = getv(p, v); s.ivl[k] = v; }
    return s;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) { if (rank == 0) fprintf(stderr, "Usage: %s input.txt [output.txt]\n", argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }

    long long N = 0; int K = 0, S = 0;
    vector<Rec> all;
    if (rank == 0) {
        ifstream fin(argv[1]);
        if (!fin) { fprintf(stderr, "Cannot open %s\n", argv[1]); MPI_Abort(MPI_COMM_WORLD, 2); }
        fin >> N >> K >> S;
        all.resize(N);
        for (long long i = 0; i < N; ++i)
            fin >> all[i].ts >> all[i].sid >> all[i].temp >> all[i].hum
                >> all[i].pres >> all[i].rain >> all[i].wind;
    }
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&S, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // record distribution (bytes), handles N not divisible by P
    vector<int> sendcnt(P), displs(P);
    long long base = N / P, rem = N % P, off = 0;
    for (int p = 0; p < P; ++p) {
        long long c = base + (p < rem ? 1 : 0);
        sendcnt[p] = (int)(c * sizeof(Rec));
        displs[p]  = (int)(off * sizeof(Rec));
        off += c;
    }
    long long myCount = base + (rank < rem ? 1 : 0);
    vector<Rec> mine(myCount);

    // ---- Phase timers (barriers give a clean wall-clock per phase) ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Phase 1: DATA DISTRIBUTION (communication) — scatter records to all ranks
    MPI_Scatterv(rank == 0 ? all.data() : nullptr, sendcnt.data(), displs.data(), MPI_BYTE,
                 mine.data(), (int)(myCount * sizeof(Rec)), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Phase 2: LOCAL COMPUTATION — the part that actually parallelises
    Stats local; local.init(S);
    for (auto& r : mine) local.add(r);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Phase 3: COMBINE (communication) — gather partial Stats and merge on rank 0
    vector<char> buf = serialize(local);
    int mylen = (int)buf.size();
    vector<int> lens(P), ldispl(P);
    MPI_Gather(&mylen, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<char> gathered;
    if (rank == 0) {
        int tot = 0;
        for (int p = 0; p < P; ++p) { ldispl[p] = tot; tot += lens[p]; }
        gathered.resize(tot);
    }
    MPI_Gatherv(buf.data(), mylen, MPI_BYTE,
                rank == 0 ? gathered.data() : nullptr, lens.data(), ldispl.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    string out;
    if (rank == 0) {
        Stats global; global.init(S);
        for (int p = 0; p < P; ++p)
            merge_into(global, deserialize(gathered.data() + ldispl[p]));
        out = format_results(global, K);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    if (rank == 0) {
        double total   = t3 - t0;
        double scatter = t1 - t0;   // data distribution (communication)
        double compute = t2 - t1;   // local computation (parallelises)
        double combine = t3 - t2;   // gather + merge (communication)
        fprintf(stderr,
                "MPI  P=%d N=%lld K=%d S=%d  time=%.6f s  scatter=%.6f compute=%.6f combine=%.6f\n",
                P, N, K, S, total, scatter, compute, combine);
        if (argc >= 3) { ofstream f(argv[2]); f << out; }
        else           { fputs(out.c_str(), stdout); }
    }

    MPI_Finalize();
    return 0;
}
