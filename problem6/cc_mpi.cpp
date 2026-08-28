// Q6 - Connected components of a large undirected graph (MPI).
//
// Algorithm: distributed label propagation accelerated by a local union-find.
//   lab[v] = current candidate component id for v (starts at v, only decreases).
//   Each round every process:
//     1. seeds a union-find with the labels known so far: union(i, lab[i]),
//     2. unions the endpoints of the edges of the vertices it owns,
//     3. takes newlab[i] = find(i)  (the root is the smallest id in the set),
//     4. MPI_Allreduce(MPI_MIN) merges what all processes learned.
//   Repeat until the label array stops changing. At the fixpoint the labels are
//   constant on every edge, so every vertex carries the minimum vertex id of its
//   component. Labels only ever shrink, so termination is guaranteed; each round
//   roughly doubles the reach of the merging, so only a handful of rounds run.
//
// Vertices (and their adjacency lists) are block-distributed with MPI_Scatterv,
// so no process stores the whole edge set.

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

// Union-find where the root of a set is always its smallest vertex id.
struct DSU {
    vector<int> p;
    void reset(int n) {
        p.resize(n);
        for (int i = 0; i < n; i++) p[i] = i;
    }
    int find(int x) {
        while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }   // path halving
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (a < b) p[b] = a; else p[a] = b;
    }
};

// Read "V" followed by V lines of "k v1 .. vk" into CSR form (rank 0 only).
static void read_graph(const char *path, int &V, vector<int> &off, vector<int> &adj) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<char> buf(sz + 1);
    sz = (long)fread(buf.data(), 1, sz, f);
    buf[sz] = 0;
    fclose(f);

    const char *s = buf.data();
    auto next_int = [&]() -> long {
        while (*s && (*s < '0' || *s > '9') && *s != '-') s++;
        long sign = 1;
        if (*s == '-') { sign = -1; s++; }
        long x = 0;
        while (*s >= '0' && *s <= '9') { x = x * 10 + (*s - '0'); s++; }
        return sign * x;
    };

    V = (int)next_int();
    off.assign(V + 1, 0);
    adj.clear();
    for (int i = 0; i < V; i++) {
        int k = (int)next_int();
        for (int j = 0; j < k; j++) adj.push_back((int)next_int());
        off[i + 1] = (int)adj.size();
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "usage: %s input.txt [output.txt]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    double t0 = MPI_Wtime();

    int V = 0;
    vector<int> off, adj;                       // full CSR, rank 0 only
    if (rank == 0) read_graph(argv[1], V, off, adj);
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Block distribution of the vertices: rank r owns [lo(r), lo(r+1)).
    int base = (P > 0) ? V / P : 0, rem = (P > 0) ? V % P : 0;
    auto lo = [&](int r) { return r * base + (r < rem ? r : rem); };
    int my_lo = lo(rank), my_n = lo(rank + 1) - my_lo;

    // Scatter the degrees, then the adjacency lists themselves.
    vector<int> vcounts(P), vdispls(P);
    for (int r = 0; r < P; r++) { vcounts[r] = lo(r + 1) - lo(r); vdispls[r] = lo(r); }

    vector<int> alldeg;
    if (rank == 0) {
        alldeg.resize(V);
        for (int i = 0; i < V; i++) alldeg[i] = off[i + 1] - off[i];
    }
    vector<int> deg(my_n);
    MPI_Scatterv(rank == 0 ? alldeg.data() : nullptr, vcounts.data(), vdispls.data(), MPI_INT,
                 deg.data(), my_n, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> loff(my_n + 1, 0);
    for (int i = 0; i < my_n; i++) loff[i + 1] = loff[i] + deg[i];
    int my_edges = loff[my_n];

    vector<int> ecounts(P, 0), edispls(P, 0);
    if (rank == 0) {
        for (int r = 0; r < P; r++) {
            ecounts[r] = off[lo(r + 1)] - off[lo(r)];
            edispls[r] = off[lo(r)];
        }
    }
    vector<int> ladj(my_edges);
    MPI_Scatterv(rank == 0 ? adj.data() : nullptr, ecounts.data(), edispls.data(), MPI_INT,
                 ladj.data(), my_edges, MPI_INT, 0, MPI_COMM_WORLD);

    off.clear(); off.shrink_to_fit();
    adj.clear(); adj.shrink_to_fit();

    MPI_Barrier(MPI_COMM_WORLD);
    double t_setup = MPI_Wtime();

    // ---- iterative min-label propagation ----
    vector<int> lab(V), newlab(V), merged(V);
    for (int i = 0; i < V; i++) lab[i] = i;

    DSU d;
    int rounds = 0;
    double comm_time = 0.0;
    while (V > 0) {
        d.reset(V);
        for (int i = 0; i < V; i++) d.unite(i, lab[i]);            // what we already know
        for (int i = 0; i < my_n; i++) {                            // our own edges
            int v = my_lo + i;
            for (int j = loff[i]; j < loff[i + 1]; j++) d.unite(v, ladj[j]);
        }
        for (int i = 0; i < V; i++) newlab[i] = d.find(i);

        double c0 = MPI_Wtime();
        MPI_Allreduce(newlab.data(), merged.data(), V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - c0;

        rounds++;
        if (merged == lab) break;      // identical on every rank, so no extra reduce needed
        lab.swap(merged);
    }

    double t_compute = MPI_Wtime();

    if (rank == 0) {
        FILE *out = stdout;
        if (argc >= 3) {
            out = fopen(argv[2], "w");
            if (!out) { fprintf(stderr, "cannot write %s\n", argv[2]); MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        for (int i = 0; i < V; i++) fprintf(out, "%d %d\n", i, lab[i]);
        if (out != stdout) fclose(out);
    }

    double t_end = MPI_Wtime();

    long long mine = my_edges, E = 0;
    MPI_Reduce(&mine, &E, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        fprintf(stderr, "P=%d V=%d E=%lld rounds=%d read+scatter=%.4fs compute=%.4fs "
                        "(comm=%.4fs) output=%.4fs total=%.4fs\n",
                P, V, E, rounds, t_setup - t0, t_compute - t_setup, comm_time,
                t_end - t_compute, t_end - t0);
    }

    MPI_Finalize();
    return 0;
}
