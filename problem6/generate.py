#!/usr/bin/env python3
"""Generate input.txt files for Q6 (connected components).

Single file:
    python3 generate.py V E [--mode MODE] [--seed S] [-o out.txt]
Whole benchmark suite (one file per size):
    python3 generate.py --suite [-d outdir]

Modes:
    random    E random undirected edges over V vertices (many small components)
    clusters  ~sqrt(V) dense blobs, a few random edges between them
    chain     one long path (worst case for label propagation: large diameter)
    grid      2D grid, one big component
    empty     no edges at all (V singleton components)
"""

import argparse
import os
import random
import sys


def gen_edges(V, E, mode, rng):
    """Return a list of (u, v) undirected edges, u != v, no duplicates."""
    edges = set()

    def add(u, v):
        if u != v:
            edges.add((min(u, v), max(u, v)))

    if mode == "empty" or V < 2:
        return []

    if mode == "chain":
        for i in range(V - 1):
            add(i, i + 1)

    elif mode == "grid":
        side = max(2, int(V ** 0.5))
        for i in range(V):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < V:
                add(i, i + 1)
            if i + side < V:
                add(i, i + side)

    elif mode == "clusters":
        nblobs = max(2, int(V ** 0.5))
        size = V // nblobs
        # dense-ish inside each blob
        per_blob = max(1, (E * 9 // 10) // nblobs)
        for b in range(nblobs):
            lo = b * size
            hi = V if b == nblobs - 1 else lo + size
            if hi - lo < 2:
                continue
            for i in range(lo + 1, hi):          # keep the blob connected
                add(i, rng.randrange(lo, i))
            for _ in range(per_blob):
                add(rng.randrange(lo, hi), rng.randrange(lo, hi))
        while len(edges) < E:                     # a few long-range edges
            add(rng.randrange(V), rng.randrange(V))

    else:  # random
        guard = 0
        while len(edges) < E and guard < 20 * E + 1000:
            add(rng.randrange(V), rng.randrange(V))
            guard += 1

    return list(edges)


def write_graph(path, V, edges):
    adj = [[] for _ in range(V)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    with open(path, "w") as f:
        f.write("%d\n" % V)
        for nbrs in adj:
            f.write(" ".join([str(len(nbrs))] + [str(x) for x in nbrs]) + "\n")
    return sum(len(a) for a in adj)


SUITE = [
    # (V, E, mode)
    (1000,   5000,    "random"),
    (10000,  50000,   "random"),
    (50000,  300000,  "clusters"),
    (100000, 1000000, "clusters"),
    (100000, 99999,   "chain"),
    # Over-spec stress case: at the assignment's own maximum (V=1e5, E=1e6) the
    # compute phase is only a few ms, so speed-up numbers would be pure noise.
    # This one gives the scaling study something to actually measure.
    (400000, 2000000, "clusters"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("V", nargs="?", type=int)
    ap.add_argument("E", nargs="?", type=int)
    ap.add_argument("--mode", default="random",
                    choices=["random", "clusters", "chain", "grid", "empty"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", default="input.txt")
    ap.add_argument("--suite", action="store_true", help="generate the benchmark suite")
    ap.add_argument("-d", "--dir", default=".", help="output directory for --suite")
    args = ap.parse_args()

    if args.suite:
        os.makedirs(args.dir, exist_ok=True)
        for V, E, mode in SUITE:
            path = os.path.join(args.dir, "input_V%d_E%d_%s.txt" % (V, E, mode))
            rng = random.Random(args.seed)
            entries = write_graph(path, V, gen_edges(V, E, mode, rng))
            print("%s  V=%d entries=%d mode=%s" % (path, V, entries, mode))
        return

    if args.V is None or args.E is None:
        ap.error("give V and E, or use --suite")
    rng = random.Random(args.seed)
    entries = write_graph(args.out, args.V, gen_edges(args.V, args.E, args.mode, rng))
    print("%s  V=%d entries=%d mode=%s" % (args.out, args.V, entries, args.mode),
          file=sys.stderr)


if __name__ == "__main__":
    main()
