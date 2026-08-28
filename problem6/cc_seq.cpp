// Q6 - Connected components, sequential reference implementation.
// Plain union-find over every edge; the root of a set is its smallest vertex id,
// so find(v) is exactly the required component id.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
using namespace std;

static vector<int> par;

static int find(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

static void unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return;
    if (a < b) par[b] = a; else par[a] = b;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s input.txt [output.txt]\n", argv[0]); return 1; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    clock_t t0 = clock();

    int V;
    if (fscanf(f, "%d", &V) != 1) { fprintf(stderr, "bad input\n"); return 1; }
    par.resize(V);
    for (int i = 0; i < V; i++) par[i] = i;

    long long E = 0;
    for (int i = 0; i < V; i++) {
        int k;
        if (fscanf(f, "%d", &k) != 1) { fprintf(stderr, "bad input at vertex %d\n", i); return 1; }
        for (int j = 0; j < k; j++) {
            int u;
            if (fscanf(f, "%d", &u) != 1) { fprintf(stderr, "bad input at vertex %d\n", i); return 1; }
            unite(i, u);
            E++;
        }
    }
    fclose(f);

    clock_t t1 = clock();

    FILE *out = stdout;
    if (argc >= 3) {
        out = fopen(argv[2], "w");
        if (!out) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    }
    for (int i = 0; i < V; i++) fprintf(out, "%d %d\n", i, find(i));
    if (out != stdout) fclose(out);

    fprintf(stderr, "sequential V=%d E=%lld time=%.4fs\n",
            V, E, double(t1 - t0) / CLOCKS_PER_SEC);
    return 0;
}
