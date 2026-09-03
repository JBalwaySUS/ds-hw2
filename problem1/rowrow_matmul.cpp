#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using namespace std;

// Core kernel : localC(localRows x p) += localA(localRows x n) * B(n x p)
// Uses the row-row / "weighted sum of B's rows" formulation.
static void multiply_rows(const vector<int>& localA, const vector<int>& B,
                          vector<long long>& localC,
                          int localRows, int n, int p) {
    for (int i = 0; i < localRows; ++i) {
        const int* arow = &localA[(size_t)i * n];
        long long* crow = &localC[(size_t)i * p];
        for (int k = 0; k < n; ++k) {
            int a = arow[k];
            if (a == 0) continue;                 // skip zero weights
            const int* brow = &B[(size_t)k * p];
            for (int j = 0; j < p; ++j)
                crow[j] += (long long)a * brow[j]; // accumulate in 64-bit (overflow-safe)
        }
    }
}

// MPI collective counts are 'int'. Fail loudly rather than silently wrapping.
static void require_int_count(size_t v, const char* what, int rank) {
    if (v > (size_t)INT_MAX) {
        if (rank == 0)
            fprintf(stderr, "Error: %s = %zu exceeds INT_MAX; MPI element counts are int.\n", what, v);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = 0, n = 0, p = 0;
    vector<int> A;   // full A on master only (row-major m x n)
    vector<int> B;   // full B on EVERY rank (row-major n x p)

    // argument parsing (every rank sees the same argv)
    bool genMode = false, writeOut = false;
    string inPath, outPath;
    unsigned seed = 12345;

    vector<string> args(argv + 1, argv + argc);
    if (!args.empty() && args[0] == "--gen") {
        genMode = true;
        if (args.size() < 4) {
            if (rank == 0) fprintf(stderr, "Usage: %s --gen m n p [seed]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        m = atoi(args[1].c_str());
        n = atoi(args[2].c_str());
        p = atoi(args[3].c_str());
        if (args.size() >= 5) seed = (unsigned)strtoul(args[4].c_str(), nullptr, 10);
    } else if (!args.empty()) {
        inPath = args[0];
        if (args.size() >= 2) { writeOut = true; outPath = args[1]; }
    } else {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <input.txt> [output.txt]\n       %s --gen m n p [seed]\n",
                    argv[0], argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // master reads or generates A and B
    if (rank == 0) {
        if (genMode) {
            mt19937 rng(seed);
            uniform_int_distribution<int> dist(-9, 9);
            A.resize((size_t)m * n);
            B.resize((size_t)n * p);
            for (auto& x : A) x = dist(rng);
            for (auto& x : B) x = dist(rng);
        } else {
            ifstream fin(inPath);
            if (!fin) { fprintf(stderr, "Cannot open %s\n", inPath.c_str()); MPI_Abort(MPI_COMM_WORLD, 2); }
            fin >> m >> n >> p;
            A.resize((size_t)m * n);
            B.resize((size_t)n * p);
            for (auto& x : A) fin >> x;
            for (auto& x : B) fin >> x;
            if (!fin) { fprintf(stderr, "Malformed input %s: expected %d + %d integers\n",
                                inPath.c_str(), m * n, n * p); MPI_Abort(MPI_COMM_WORLD, 2); }
        }
    }

    // Broadcast the three dimensions. This is 3 ints of metadata, not part of the
    // data-movement phase, so it stays outside the timed region.
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // row distribution (handles m NOT divisible by P)
    // First 'rem' processes get one extra row.
    vector<int> rows(size), sendA(size), dispA(size), recvC(size), dispC(size);
    int base = m / size, rem = m % size;
    size_t offA = 0, offC = 0;
    for (int i = 0; i < size; ++i) {
        rows[i] = base + (i < rem ? 1 : 0);
        require_int_count((size_t)rows[i] * n, "rows*n", rank);
        require_int_count((size_t)rows[i] * p, "rows*p", rank);
        require_int_count(offA + (size_t)rows[i] * n, "A displacement", rank);
        require_int_count(offC + (size_t)rows[i] * p, "C displacement", rank);
        sendA[i] = rows[i] * n;          // ints of A for rank i
        recvC[i] = rows[i] * p;          // long longs of C from rank i
        dispA[i] = (int)offA; offA += (size_t)sendA[i];
        dispC[i] = (int)offC; offC += (size_t)recvC[i];
    }
    int localRows = rows[rank];

    // Buffer allocation is local work, so it is done before the clock starts.
    require_int_count((size_t)n * p, "n*p", rank);
    if (rank != 0) B.resize((size_t)n * p);
    vector<int>       localA((size_t)localRows * n);
    vector<long long> localC((size_t)localRows * p, 0);
    vector<long long> C;
    if (rank == 0) C.resize((size_t)m * p);

    // timed region : broadcast B -> scatter A -> compute -> gather C
    // The broadcast of B is genuine parallel-phase communication and is therefore
    // measured, not excluded.
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Bcast(B.data(), n * p, MPI_INT, 0, MPI_COMM_WORLD);
    double tb = MPI_Wtime();

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendA.data(), dispA.data(), MPI_INT,
                 localA.data(), localRows * n, MPI_INT,
                 0, MPI_COMM_WORLD);
    double ts = MPI_Wtime();

    multiply_rows(localA, B, localC, localRows, n, p);
    double tc = MPI_Wtime();

    MPI_Gatherv(localC.data(), localRows * p, MPI_LONG_LONG,
                rank == 0 ? C.data() : nullptr, recvC.data(), dispC.data(), MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Per-phase cost = slowest rank for that phase, so the breakdown reflects the
    // critical path rather than rank 0's private view.
    double mine[4] = { tb - t0, ts - tb, tc - ts, t1 - tc };
    double worst[4];
    MPI_Reduce(mine, worst, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // output
    if (rank == 0) {
        fprintf(stderr, "P=%d  m=%d n=%d p=%d  time=%.6f s"
                        "  bcastB=%.6f scatterA=%.6f compute=%.6f gatherC=%.6f\n",
                size, m, n, p, t1 - t0, worst[0], worst[1], worst[2], worst[3]);
        if (!genMode) {
            string buf;
            buf.reserve((size_t)m * p * 4);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < p; ++j) {
                    buf += to_string(C[(size_t)i * p + j]);
                    buf += (j + 1 < p ? ' ' : '\n');
                }
            if (writeOut) { ofstream fout(outPath); fout << buf; }
            else          { fputs(buf.c_str(), stdout); }
        }
    }

    MPI_Finalize();
    return 0;
}
