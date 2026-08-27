#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "Usage: %s m n p seed [outfile]\n", argv[0]); return 1; }
    int m = atoi(argv[1]), n = atoi(argv[2]), p = atoi(argv[3]);
    unsigned seed = (unsigned)strtoul(argv[4], nullptr, 10);
    ostream* out = &cout; ofstream fout;
    if (argc >= 6) { fout.open(argv[5]); out = &fout; }
    mt19937 rng(seed);
    uniform_int_distribution<int> d(-9, 9);
    (*out) << m << ' ' << n << ' ' << p << '\n';
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) (*out) << d(rng) << (j + 1 < n ? ' ' : '\n');
    for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) (*out) << d(rng) << (j + 1 < p ? ' ' : '\n');
    return 0;
}
