#!/usr/bin/env python3
"""Generate random input graphs for Q6 (connected components).

Single input file:
    python3 gen_graph.py V E [--seed S] [-o out.txt]

Four-case benchmark suite:
    python3 gen_graph.py --suite [-d outdir]

The suite's small case is the exact sample input from task.md. The other three
cases are reproducible random undirected graphs.
"""

import argparse
import os
import random
import sys


MAX_VERTICES = 100_000
MAX_EDGES = 1_000_000

# The three undirected edges from the sample input in task.md. In this order,
# write_graph also reproduces the sample adjacency lists exactly.
SAMPLE_EDGES = [(0, 1), (2, 3), (2, 4)]

# One input for each row of the timing table. Small is handled specially so
# that it is the task's sample; all remaining cases use gen_edges.
SUITE = [
    # (name, V, E)
    ("small", 5, 3),
    ("medium", 500, 2500),
    ("large", 5000, 25000),
    ("very_large", 100000, 1000000),
]


def validate_size(vertices, edge_count):
    """Validate a requested graph against the assignment constraints."""
    if not 1 <= vertices <= MAX_VERTICES:
        raise ValueError("V must be between 1 and %d" % MAX_VERTICES)
    if not 0 <= edge_count <= MAX_EDGES:
        raise ValueError("E must be between 0 and %d" % MAX_EDGES)

    possible_edges = vertices * (vertices - 1) // 2
    if edge_count > possible_edges:
        raise ValueError(
            "E=%d is impossible for a simple graph with V=%d (maximum %d)"
            % (edge_count, vertices, possible_edges)
        )


def gen_edges(vertices, edge_count, rng):
    """Return edge_count random undirected edges without loops or duplicates."""
    validate_size(vertices, edge_count)
    edges = set()
    while len(edges) < edge_count:
        u = rng.randrange(vertices)
        v = rng.randrange(vertices)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return list(edges)


def write_graph(path, vertices, edges):
    adjacency = [[] for _ in range(vertices)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    with open(path, "w") as output:
        output.write("%d\n" % vertices)
        for neighbors in adjacency:
            output.write(
                " ".join([str(len(neighbors))] + [str(v) for v in neighbors])
                + "\n"
            )


def generate_suite(output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    for index, (name, vertices, edge_count) in enumerate(SUITE):
        if name == "small":
            edges = SAMPLE_EDGES
        else:
            # Give each size its own deterministic random stream.
            edges = gen_edges(vertices, edge_count, random.Random(seed + index))

        path = os.path.join(output_dir, "input_%s.txt" % name)
        write_graph(path, vertices, edges)
        print("%s  V=%d E=%d" % (path, vertices, len(edges)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("V", nargs="?", type=int)
    parser.add_argument("E", nargs="?", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--out", default="input.txt")
    parser.add_argument(
        "--suite", action="store_true", help="generate the four benchmark inputs"
    )
    parser.add_argument("-d", "--dir", default=".", help="output directory for --suite")
    args = parser.parse_args()

    if args.suite:
        if args.V is not None or args.E is not None:
            parser.error("V and E cannot be used with --suite")
        generate_suite(args.dir, args.seed)
        return

    if args.V is None or args.E is None:
        parser.error("give V and E, or use --suite")
    try:
        edges = gen_edges(args.V, args.E, random.Random(args.seed))
    except ValueError as error:
        parser.error(str(error))
    write_graph(args.out, args.V, edges)
    print("%s  V=%d E=%d" % (args.out, args.V, len(edges)), file=sys.stderr)


if __name__ == "__main__":
    main()
