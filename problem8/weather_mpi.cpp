#include <mpi.h>
#include "weather_core.h"
using namespace std;

// Combining partial results used to be: serialise each rank's Stats to bytes,
// MPI_Gatherv them all to rank 0, and merge_into() them one at a time. That is
// O(P) work on a single rank, and because the interval histogram holds one entry
// per distinct 60-second bucket (~N/4), it also moved O(P*N) bytes — so the
// combine phase grew with P and made the program slower the more ranks it got.
//
// Everything below is instead a tree-based MPI collective: O(log P) depth, and
// rank 0 never sees more than one array's worth of data.

static void die(int rank, const char* msg) {
    if (rank == 0) fprintf(stderr, "Error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 3);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) { if (rank == 0) fprintf(stderr, "Usage: %s input.txt [output.txt]\n", argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }

    long long N = 0; int K = 0, S = 0;
    vector<Rec> all;
    double parse = 0.0;              // rank 0's file read; reported, not timed in
    if (rank == 0) {
        double pr0 = MPI_Wtime();
        ifstream fin(argv[1]);
        if (!fin) { fprintf(stderr, "Cannot open %s\n", argv[1]); MPI_Abort(MPI_COMM_WORLD, 2); }
        fin >> N >> K >> S;
        all.resize(N);
        for (long long i = 0; i < N; ++i)
            fin >> all[i].ts >> all[i].sid >> all[i].temp >> all[i].hum
                >> all[i].pres >> all[i].rain >> all[i].wind;
        if (!fin) { fprintf(stderr, "Malformed input %s: expected %lld records\n", argv[1], N); MPI_Abort(MPI_COMM_WORLD, 2); }
        parse = MPI_Wtime() - pr0;
    }
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&S, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Record distribution (bytes), handles N not divisible by P.
    // MPI counts/displacements are int, so guard the byte arithmetic explicitly.
    if (N > (long long)(INT_MAX / sizeof(Rec)))
        die(rank, "N too large: scatter byte count would overflow int (use parallel I/O instead)");
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

    // Phase 3: COMBINE — tree reductions, no gather-and-merge on rank 0
    Stats g; g.init(S);

    // (a) scalar accumulators: one reduction per operator
    double lsum[5] = { local.sum_temp, local.sum_hum, local.sum_pres, local.sum_rain, local.sum_wind };
    double lmin[3] = { local.min_temp, local.min_hum, local.min_pres };
    double lmax[5] = { local.max_temp, local.max_hum, local.max_pres, local.max_rain, local.max_wind };
    long long lcnt[2] = { local.total, local.extreme };
    // Initialised so non-root ranks (where MPI_Reduce writes nothing) hold
    // defined values; only rank 0 ever reads them.
    double gsum[5] = {0,0,0,0,0}, gmin[3] = {0,0,0}, gmax[5] = {0,0,0,0,0};
    long long gcnt[2] = {0,0};
    MPI_Reduce(lsum, gsum, 5, MPI_DOUBLE,    MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(lmin, gmin, 3, MPI_DOUBLE,    MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(lmax, gmax, 5, MPI_DOUBLE,    MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(lcnt, gcnt, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // (b) per-station arrays: three SUM reductions of length S
    if (S > 0) {
        MPI_Reduce(local.st_count.data(),   g.st_count.data(),   S, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local.st_sumtemp.data(), g.st_sumtemp.data(), S, MPI_DOUBLE,    MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local.st_sumrain.data(), g.st_sumrain.data(), S, MPI_DOUBLE,    MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // (c) interval histogram: agree on one global bucket range, then SUM-reduce
    //     the dense array. Ranks holding no records contribute identity sentinels.
    long long llo = local.ivl.empty() ? LLONG_MAX : local.ivl.lo();
    long long lhi = local.ivl.empty() ? LLONG_MIN : local.ivl.hi();
    long long glo, ghi;
    MPI_Allreduce(&llo, &glo, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lhi, &ghi, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (glo < ghi) {
        long long span = ghi - glo;
        if (span > IVL_MAX_BUCKETS || span > INT_MAX)
            die(rank, "interval range too large for a dense histogram");
        local.ivl.rebase(glo, span);
        g.ivl.base = glo; g.ivl.c.assign((size_t)span, 0);
        MPI_Reduce(local.ivl.c.data(), g.ivl.c.data(), (int)span, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // (d) hottest / coldest: P tiny structs, picked with the shared comparators
    //     so the tie-break rules stay in exactly one place.
    vector<Meas> hots(rank == 0 ? P : 0), colds(rank == 0 ? P : 0);
    MPI_Gather(&local.hot,  sizeof(Meas), MPI_BYTE, hots.data(),  sizeof(Meas), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local.cold, sizeof(Meas), MPI_BYTE, colds.data(), sizeof(Meas), MPI_BYTE, 0, MPI_COMM_WORLD);

    string out;
    if (rank == 0) {
        g.total = gcnt[0]; g.extreme = gcnt[1];
        g.sum_temp = gsum[0]; g.sum_hum = gsum[1]; g.sum_pres = gsum[2]; g.sum_rain = gsum[3]; g.sum_wind = gsum[4];
        g.min_temp = gmin[0]; g.min_hum = gmin[1]; g.min_pres = gmin[2];
        g.max_temp = gmax[0]; g.max_hum = gmax[1]; g.max_pres = gmax[2]; g.max_rain = gmax[3]; g.max_wind = gmax[4];
        for (int p = 0; p < P; ++p) {
            if (is_hotter(hots[p],  g.hot))  g.hot  = hots[p];
            if (is_colder(colds[p], g.cold)) g.cold = colds[p];
        }
        out = format_results(g, K);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    if (rank == 0) {
        fprintf(stderr,
                "MPI  P=%d N=%lld K=%d S=%d  time=%.6f s  scatter=%.6f compute=%.6f combine=%.6f parse=%.6f\n",
                P, N, K, S, t3 - t0, t1 - t0, t2 - t1, t3 - t2, parse);
        if (argc >= 3) { ofstream f(argv[2]); f << out; }
        else           { fputs(out.c_str(), stdout); }
    }

    MPI_Finalize();
    return 0;
}
