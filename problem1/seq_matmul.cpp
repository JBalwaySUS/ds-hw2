#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s input.txt [output.txt]\n", argv[0]); return 1; }
    ifstream fin(argv[1]);
    if (!fin) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 2; }
    int m, n, p; fin >> m >> n >> p;
    vector<int> A((size_t)m * n), B((size_t)n * p);
    for (auto& x : A) fin >> x;
    for (auto& x : B) fin >> x;

    vector<long long> C((size_t)m * p, 0);
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < n; ++k) {
            long long a = A[(size_t)i * n + k];
            if (!a) continue;
            const int* brow = &B[(size_t)k * p];
            long long* crow = &C[(size_t)i * p];
            for (int j = 0; j < p; ++j) crow[j] += a * brow[j];
        }

    ostream* out = &cout; ofstream fout;
    if (argc >= 3) { fout.open(argv[2]); out = &fout; }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            (*out) << C[(size_t)i * p + j] << (j + 1 < p ? ' ' : '\n');
    return 0;
}
