#include <mpi.h>
#include <bits/stdc++.h>
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
        }
    }

    //broadcast dimensions, then broadcast full B 
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) B.resize((size_t)n * p);
    MPI_Bcast(B.data(), n * p, MPI_INT, 0, MPI_COMM_WORLD);

    // row distribution (handles m NOT divisible by P) 
    // First 'rem' processes get one extra row.
    vector<int> rows(size), sendA(size), dispA(size), recvC(size), dispC(size);
    int base = m / size, rem = m % size, offA = 0, offC = 0;
    for (int i = 0; i < size; ++i) {
        rows[i]  = base + (i < rem ? 1 : 0);
        sendA[i] = rows[i] * n;          // ints of A for rank i
        recvC[i] = rows[i] * p;          // long longs of C from rank i
        dispA[i] = offA; offA += sendA[i];
        dispC[i] = offC; offC += recvC[i];
    }
    int localRows = rows[rank];
    vector<int>       localA((size_t)localRows * n);
    vector<long long> localC((size_t)localRows * p, 0);

    // timed region : scatter A -> compute -> gather C 
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendA.data(), dispA.data(), MPI_INT,
                 localA.data(), localRows * n, MPI_INT,
                 0, MPI_COMM_WORLD);

    multiply_rows(localA, B, localC, localRows, n, p);

    vector<long long> C;
    if (rank == 0) C.resize((size_t)m * p);
    MPI_Gatherv(localC.data(), localRows * p, MPI_LONG_LONG,
                rank == 0 ? C.data() : nullptr, recvC.data(), dispC.data(), MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // output
    if (rank == 0) {
        fprintf(stderr, "P=%d  m=%d n=%d p=%d  time=%.6f s\n", size, m, n, p, t1 - t0);
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
