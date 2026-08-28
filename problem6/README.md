# Q6 — Connected Components of a Large Graph (MPI)

Assigns every vertex of an undirected graph the **minimum vertex id in its
connected component**. Vertices and their adjacency lists are distributed across
the MPI processes; the component ids are agreed on by message passing.

## Files
- `cc_mpi.cpp`   — MPI implementation (the parallel program).
- `cc_seq.cpp`   — sequential union-find implementation (correctness ground truth).
- `generate.py`  — generates `input.txt` files of various sizes/shapes.
- `submit_q6.sh` — SLURM script: compiles both programs with profiling enabled,
  generates the graphs, runs P = 1, 2, 4, 8 over every input size, verifies each
  run against the sequential result, and dumps the mpiP/gprof reports into the log.

## Algorithm
Distributed label propagation accelerated by a local union-find.

`lab[v]` is v's current candidate component id, starting at `v`. Each round,
every process:

1. seeds a union-find with what is already known — `union(i, lab[i])` for all `i`;
2. unions the endpoints of the edges of the vertices **it owns**;
3. sets `newlab[i] = find(i)` (the root of a set is always its smallest id);
4. `MPI_Allreduce(MPI_MIN)` merges what every process learned.

The loop stops when the label array stops changing. Labels only ever decrease
and always name a vertex in the same true component, so termination is
guaranteed; at the fixpoint the labels are constant across every edge, which
makes each label the component minimum. Because step 1 folds in *all* previously
merged information, the reach of the merging roughly doubles per round — in
practice **2–4 rounds**, even for a long chain (the worst case for plain
one-hop propagation) split across 8 processes.

Cost per round: `O(V + E_local·α)` compute, one `Allreduce` of `V` ints.

- Vertices are block-distributed; adjacency lists are handed out with
  `MPI_Scatterv`, so no rank stores the whole edge set.
- The `V`-sized label array is replicated on every rank — at `V ≤ 10^5` that is
  400 KB, which is what makes the single-collective-per-round design worthwhile.
- Adjacency lists need not be symmetric, and self-loops/duplicate edges are fine.

## Input / output
Input (`V`, then `V` lines of `k v1 … vk`) and output (`V` lines of
`vertex_id component_id`, sorted by vertex id) exactly as in `task.md`.
Timing information goes to **stderr**, never stdout.

## Compile
```bash
module load openmpi/4.1.5          # or: module load hpcx-2.7.0/hpcx-ompi
mpicxx -O2 -std=c++17 -o cc_mpi cc_mpi.cpp
g++    -O2 -std=c++17 -o cc_seq cc_seq.cpp
```

With profiling enabled (mpiP ships with the openmpi module):
```bash
mpicxx -g -O2 -std=c++17 -o cc_mpi cc_mpi.cpp -lmpiP -lm -lbfd -liberty -lunwind -ldl
g++    -g -O2 -pg -std=c++17 -o cc_seq cc_seq.cpp
export MPIP="-t 5.0 -k 2"
```

## Run
```bash
mpirun -np 4 --oversubscribe ./cc_mpi input.txt            # print to stdout
mpirun -np 4 --oversubscribe ./cc_mpi input.txt out.txt    # write to out.txt
./cc_seq input.txt out_seq.txt                             # sequential reference
diff out_seq.txt out.txt                                   # verify
```

## Generate inputs
```bash
python3 generate.py 100000 1000000 --mode clusters -o input.txt
python3 generate.py --suite -d data        # the whole benchmark suite
```
Modes: `random`, `clusters`, `chain` (large diameter — the hard case), `grid`,
`empty`. `--seed` makes every graph reproducible (default 42).

## Benchmark
```bash
sbatch submit_q6.sh
```
Everything — timings, PASS/FAIL per run, the mpiP report after each run and a
`gprof` profile of the sequential program — goes into `q6_<jobid>.log`.

## A note on the sizes
At the assignment's stated maximum (`V = 10^5`, `E = 10^6`) the whole
computation is only ~10 ms, so speed-up over `P` measures little except
collective latency. The suite therefore also contains one deliberately
over-spec case (`V = 4·10^5`, `E = 2·10^6`) so the scaling study has something
real to measure. The in-spec sizes are all still there.

## Verification status
- `cc_seq` reproduces the sample input/output in `task.md` exactly.
- `cc_mpi` and `cc_seq` agree on the whole generated suite and on the edge cases
  (`E = 0`, `V = 1`, chains, grids, asymmetric adjacency lists).
- The distributed loop (block split + local union-find + `Allreduce`-MIN) was
  additionally checked for `P = 1, 2, 4, 8` against BFS ground truth on 400
  random graphs and on chains up to 1000 vertices.
