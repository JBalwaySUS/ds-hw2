# Distributed Systems — Home Work 2

**Team Members:** \
Shubham Sunny (2025201025) \
Anirudh Sankar (2023111024)

---

# Distributed Algorithms: Q1 — Distributed Matrix Multiplication (Row-Row Method)

## 1. Problem

Given two integer matrices `A` (dimension *m × n*) and `B` (dimension *n × p*),
compute the product `C = A × B` (dimension *m × p*) using MPI and the **Row-Row
method**.

In the Row-Row method each row of `C` is expressed as a *weighted sum of the rows
of `B`*, where the weights are the entries of the corresponding row of `A`:

```
c_i = a_i[0]·B[0,:] + a_i[1]·B[1,:] + ... + a_i[n-1]·B[n-1,:]
```

## 2. Approach

The work is divided across `P` MPI processes as follows:

1. **Distribute A row-wise.** The master splits `A` into row-blocks of roughly
   *m / P* rows and scatters one block to each process.
2. **Broadcast B in full.** Every process receives a complete copy of `B`,
   because computing any row of `C` needs every row of `B`.
3. **Local computation.** Each process computes its own rows of `C` independently
   as the weighted sum described above. There is **no worker-to-worker
   communication** — the row-blocks are independent.
4. **Gather C.** The master collects every process's row-block and reassembles the
   full result matrix `C`.

Because *m* need not be divisible by *P*, the distribution uses `MPI_Scatterv`
and `MPI_Gatherv` (variable-sized blocks) rather than the fixed-size `MPI_Scatter`
/ `MPI_Gather`. The first `m mod P` processes each take one extra row, so the row
counts differ by at most one and the load stays balanced.

The core MPI primitives used are: `MPI_Bcast` (dimensions and `B`),
`MPI_Scatterv` (rows of `A`), and `MPI_Gatherv` (rows of `C`).


## 3. Implementation details

**Input format** (`m n p` header, then both matrices in row-major order):

```
m n p
<m*n integers : matrix A>
<n*p integers : matrix B>
```

**Output:** the *m × p* result matrix `C`, printed as *m* lines of *p*
space-separated integers on **stdout**. Timing information is written to
**stderr** only, so the required program output on stdout is never polluted by
diagnostics.

**Overflow safety.** Although the inputs are integers, the accumulated products
can exceed 32-bit range for large *n*. All accumulation is therefore done in
64-bit (`long long`), and `C` is gathered as `MPI_LONG_LONG`.

**Edge cases handled.** *m* not divisible by *P*; *m = 1*, *n = 1*, *p = 1*; tall
(*m ≫ n*) and wide (*n ≫ m*) matrices; and *m < P* (in which case the surplus
processes simply receive zero rows and do nothing). A zero entry of `A`
short-circuits the inner loop as a small optimisation.

**Files**

| File | Purpose |
|------|---------|
| `rowrow_matmul.cpp` | MPI Row-Row implementation (the deliverable program) |
| `seq_matmul.cpp` | Sequential reference used to verify correctness |
| `gen_matrix.cpp` | Reproducible input generator (fixed seed) |
| `run_q1.sh` | SLURM batch script (`sbatch run_q1.sh`): compile → correctness → benchmark → plots |
| `run_q1_cores.sh` | Control experiment: same benchmark on 8 *physical* cores (§7.5) |
| `plot_speedup.py` | Turns `results/bench.txt` into the speed-up / efficiency / phase-breakdown plots |
| `tests/` | 16-case correctness suite + `correctness_results.txt` |
| `results/` | Raw timings, formatted benchmark table, plots, and the raw SLURM job log |


## 4. Compilation and execution

```bash
module load hpcx-2.7.0/hpcx-ompi
mpicxx -O2 -std=c++17 -o rowrow rowrow_matmul.cpp
g++    -O2 -std=c++17 -o seq    seq_matmul.cpp
g++    -O2 -std=c++17 -o gen    gen_matrix.cpp
```

Run on an allocated compute node:

```bash
mpirun -np 4 ./rowrow input.txt            # print C to stdout
mpirun -np 4 ./rowrow input.txt out.txt    # write C to out.txt
mpirun -np 4 ./rowrow --gen 1024 1024 1024 42   # generate internally, time only
```

On this cluster two launch settings are required: `--oversubscribe` (the debug
node exposes fewer cores than 8, so P=8 must share cores) and the environment
variable `OMPI_MCA_coll_hcoll_enable=0` (silences a harmless HCA/hcoll hardware
warning). Both are set inside `run_q1.sh` and `tests/run_tests.sh`, so the whole
pipeline runs with a single `sbatch run_q1.sh` from the `Q1/` directory.

`run_q1.sh` also records `nproc`, `SLURM_CPUS_ON_NODE` and `lscpu` output at the
top of its log, so the core count that the scaling analysis in §8 depends on is
documented rather than asserted.


## 5. Correctness verification

Correctness is verified by comparing the MPI output, **byte-for-byte**, against
the sequential reference (`seq_matmul.cpp`) for every process count P = 1, 2, 4, 8.
A dedicated 16-case suite (`tests/`) covers all required shapes and edge cases:

| Case | Shape | Stresses |
|------|-------|----------|
| PDF example 1 | 3×2×3 | Known textbook answer |
| PDF example 2 | 4×2×2 | Known answer, uneven split |
| one_by_one | 1×1×1 | Smallest possible |
| identity | 4×4×4 | A×I = A |
| zeros | 4×4×3 | Zero-skip path |
| even_split | 8×4×5 | m divisible by 2/4/8 |
| uneven_split | 7×5×4 | m not divisible by 2/4 |
| single_row | 1×6×5 | m = 1 |
| n_equals_1 | 5×1×4 | n = 1 |
| p_equals_1 | 5×4×1 | p = 1 |
| tall | 30×3×4 | m ≫ n |
| wide | 3×30×4 | n ≫ m |
| fewer_rows_than_P | 3×5×5 | m < P (idle ranks) |
| odd_dims | 13×11×7 | Awkward prime dimensions |
| square | 10×10×10 | Plain square |
| medium | 100×80×60 | Larger case |

**Result:** `TOTAL: 64 passed, 0 failed — ALL TESTS PASSED`
(16 cases × 4 process counts), from job 83598 on the cluster. Full log:
`tests/correctness_results.txt`; the unedited SLURM output is in
`results/q1_83598.out`. The two PDF examples reproduce the exact matrices given
in the assignment.


## 6. Benchmark methodology

- **Environment:** RCE SLURM cluster (`rce.iiit.ac.in`), `debug` partition,
  single compute node (`node07`), OpenMPI via HPC-X 2.7.0, compiled with
  `mpicxx -O2 -std=c++17`. Job 83598.
- **Node hardware:** 2 × Intel Xeon Gold 5317 @ 3.00 GHz, 12 cores per socket,
  2 threads per core (48 logical CPUs on the node). **The job was allocated 8 of
  those logical CPUs** (`nproc = 8`, `SLURM_CPUS_ON_NODE = 8`). Because the CPUs
  are SMT siblings, 8 logical CPUs correspond to **4 physical cores**. This
  number matters for every scaling conclusion below, so `run_q1.sh` records
  `nproc`, `SLURM_CPUS_ON_NODE` and `lscpu` at the top of its log rather than
  leaving it to be assumed.
- **Inputs:** square matrices of size 256, 512 and 1024 (i.e. *m = n = p*), plus
  one deliberately skewed, non-P-divisible case (1023 × 512 × 2048) so the
  benchmark also exercises the shapes named in the input constraints. All are
  generated internally with a fixed seed (42) via `--gen`, so timings exclude
  file I/O and are reproducible on a given machine.
- **Process counts:** P = 1, 2, 4, 8, launched with `mpirun --oversubscribe`.
- **Timed region:** the *whole* parallel phase, measured with `MPI_Wtime` between
  two `MPI_Barrier`s — **broadcast of `B`**, scatter of `A`, local multiplication,
  and gather of `C`. The broadcast is genuine parallel-phase communication (it
  costs essentially nothing at P = 1 but real time at P = 8), so excluding it
  would flatter the speed-up; it is therefore measured, not skipped. Only the
  three-integer dimension broadcast and the local buffer allocation sit outside
  the clock.
- **Phase breakdown:** each run additionally reports the four phases separately.
  Each phase is reduced with `MPI_MAX` across ranks, so each figure is the
  worst-case cost of that phase — the critical path, not rank 0's private view.
  This turns the communication-vs-computation discussion below into a *measured*
  split rather than an inferred one. One consequence worth stating: because
  different ranks can be the slowest in different phases, the four maxima sum to
  slightly *more* than the wall-clock total (e.g. 0.004775 s of phases against a
  0.003802 s total at 256³, P = 8). The sum is an upper bound on the critical
  path, not an exact decomposition; all percentages quoted in §8 use the
  measured wall-clock total as the denominator.
- **Control experiment.** The main run gives each rank one SMT thread, so at
  P = 8 two ranks share one physical core. To separate *hardware allocation*
  from *algorithm*, the benchmark was repeated with `--cpus-per-task=2`
  (`run_q1_cores.sh`, job 83608), which allocates 16 logical CPUs = **8 full
  physical cores**, one per rank, with `mpirun --bind-to core`. Comparing the two
  isolates the cause of the P = 8 plateau instead of speculating about it (§7.5).

Raw timings are in `results/bench.txt` and `results/bench_physical_cores.txt`
(the exact program output, consumed by `plot_speedup.py`). A formatted version
with speed-up, efficiency and the phase table is regenerated automatically into
`results/benchmark_results.txt`. The unedited SLURM logs for both jobs are kept
in `results/q1_83598.out` and `results/q1_cores_83608.out`.


## 7. Results

All figures below come from job 83598 (`results/bench.txt`), on 8 logical CPUs
= 4 physical cores. §7.5 adds the 8-physical-core control run (job 83608).

### 7.1 Runtime (seconds)

| Input size | P=1 | P=2 | P=4 | P=8 |
|------------|:---:|:---:|:---:|:---:|
| 256×256×256    | 0.009494 | 0.005139 | 0.003528 | 0.003802 |
| 512×512×512    | 0.073175 | 0.037434 | 0.021489 | 0.024270 |
| 1024×1024×1024 | 0.578302 | 0.291111 | 0.152106 | 0.157815 |
| 1023×512×2048  | 0.574840 | 0.290470 | 0.152538 | 0.160441 |

The last row is deliberately skewed *and* has *m* = 1023 not divisible by 2, 4
or 8, so the benchmark exercises the awkward shape and the `Scatterv` remainder
path at full scale, not only in the small correctness tests.

### 7.2 Speed-up  S(P) = T₁ / T_P

| Input size | P=1 | P=2 | P=4 | P=8 |
|------------|:---:|:---:|:---:|:---:|
| 256×256×256    | 1.00 | 1.85 | 2.69 | 2.50 |
| 512×512×512    | 1.00 | 1.95 | 3.41 | 3.02 |
| 1024×1024×1024 | 1.00 | 1.99 | 3.80 | 3.66 |
| 1023×512×2048  | 1.00 | 1.98 | 3.77 | 3.58 |

![Speed-up vs P](problem1/results/speedup.png)

*Figure 1 — Speed-up S(P) vs number of processes. Dashed line = ideal linear
speed-up. Scaling is near-ideal up to P = 4 (the physical core count) and then
turns over.*

### 7.3 Efficiency  E(P) = S(P) / P

| Input size | P=1 | P=2 | P=4 | P=8 |
|------------|:---:|:---:|:---:|:---:|
| 256×256×256    | 1.00 | 0.92 | 0.67 | 0.31 |
| 512×512×512    | 1.00 | 0.98 | 0.85 | 0.38 |
| 1024×1024×1024 | 1.00 | 0.99 | 0.95 | 0.46 |
| 1023×512×2048  | 1.00 | 0.99 | 0.94 | 0.45 |

![Efficiency vs P](problem1/results/efficiency.png)

*Figure 2 — Parallel efficiency E(P) vs number of processes. At every P,
efficiency rises with problem size.*

### 7.4 Phase breakdown

Total time split into broadcast of `B`, scatter of `A`, local computation and
gather of `C` (slowest rank per phase), for 1024×1024×1024:

| P | Bcast B | Scatter A | Compute | Gather C | Wall clock |
|:-:|--------:|----------:|--------:|---------:|-----------:|
| 1 | 0.000000 | 0.000470 | 0.576816 | 0.001016 | 0.578302 |
| 2 | 0.001200 | 0.000591 | 0.288226 | 0.001198 | 0.291111 |
| 4 | 0.004034 | 0.000944 | 0.144480 | 0.003519 | 0.152106 |
| 8 | 0.009993 | 0.001768 | 0.143774 | 0.005199 | 0.157815 |

![Phase breakdown](problem1/results/phase_breakdown.png)

*Figure 3 — The same algorithm at the two extremes of problem size. At 256³ the
red gather block grows until communication is most of the runtime; at 1024³ the
green compute block still dominates at every P. Bars are annotated with the
measured wall-clock total (see the `MPI_MAX` caveat in §6).*

### 7.5 Control experiment: SMT threads vs physical cores

Identical benchmark, the only change being `--cpus-per-task=2` so each rank owns
a full physical core instead of one SMT sibling (job 83608):

| 1024×1024×1024 | P=1 | P=2 | P=4 | P=8 |
|----------------|:---:|:---:|:---:|:---:|
| Total, 4 cores / 8 SMT threads | 0.578302 | 0.291111 | 0.152106 | **0.157815** |
| Total, 8 physical cores        | 0.577713 | 0.293807 | 0.160368 | **0.096768** |
| Speed-up, 8 SMT threads        | 1.00 | 1.99 | 3.80 | **3.66** |
| Speed-up, 8 physical cores     | 1.00 | 1.97 | 3.60 | **5.97** |

The skewed case behaves identically: S(8) = 3.58 on SMT threads, **5.99** on
physical cores.

![SMT vs physical cores](problem1/results/smt_vs_cores.png)

*Figure 4 — The P = 8 turnover is a property of the CPU allocation, not of the
algorithm. Given eight real cores the same binary keeps scaling.*


## 8. Analysis: communication vs computation

### 8.1 The computation scales exactly as the row decomposition predicts

Isolating the compute phase removes all communication and shows what the
partitioning alone achieves:

| Input size | P=1 | P=2 | P=4 | P=8 |
|------------|:---:|:---:|:---:|:---:|
| 256×256×256    | 1.00 | 2.01 | 3.97 | 3.97 |
| 512×512×512    | 1.00 | 1.99 | 3.98 | 4.04 |
| 1024×1024×1024 | 1.00 | 2.00 | 3.99 | 4.01 |
| 1023×512×2048  | 1.00 | 2.00 | 3.99 | 3.96 |

*Compute-phase speed-up.* This is the central result. The row-blocks are fully
independent, so the computation scales **linearly and almost perfectly** — 3.97
to 3.99 out of an ideal 4.00 at P = 4, on every input shape including the skewed,
non-divisible one. Load balance is therefore working: giving the first *m* mod *P*
ranks one extra row keeps the row counts within one of each other, and no rank
waits noticeably for another.

At P = 8 the compute speed-up stays pinned at ≈ 4.0 on all four inputs. Four
independent measurements landing on the same ceiling is not a coincidence: the
job held **4 physical cores**, and ranks 5–8 could only run on SMT siblings that
share the same execution units. Dense matrix multiply is FP/SIMD-throughput
bound, which is precisely the workload SMT cannot help. §7.5 confirms this
directly — given 8 real cores, the same binary reaches S(8) = 5.97.

### 8.2 The communication grows with P and shrinks with problem size

Communication = broadcast of `B` + scatter of `A` + gather of `C`, as a fraction
of wall-clock time:

| Input size | P=1 | P=2 | P=4 | P=8 |
|------------|:---:|:---:|:---:|:---:|
| 256×256×256    | 0.6 % | 9.0 % | 35.6 % | **63.1 %** |
| 512×512×512    | 0.3 % | 2.4 % | 19.3 % | 33.6 % |
| 1024×1024×1024 | 0.3 % | 1.0 % | 5.6 %  | 10.7 % |
| 1023×512×2048  | 0.4 % | 1.5 % | 6.3 %  | 12.2 % |

Two trends, both explained by the algorithm's structure:

**Along a row (increasing P):** communication cost rises roughly linearly. The
broadcast of `B` goes 0.0012 → 0.0040 → 0.0100 s at 1024³ as P goes 2 → 4 → 8,
and the gather of `C` follows the same pattern. Every additional rank is another
recipient of the full *n × p* copy of `B` and another contributor to the gather,
while the compute each rank does *shrinks* as 1/P. So the ratio moves against us
at exactly the rate the model predicts.

**Down a column (increasing size):** communication cost is a *surface* quantity
and computation is a *volume* one. The transferred data is *n·p* (broadcast)
+ *m·n* (scatter) + *m·p* (gather) — quadratic — while the work is *m·n·p*
multiply-adds — cubic. Multiplying the dimensions by 4 (256 → 1024) multiplies
communication by ≈ 16 but computation by ≈ 64, so the model predicts the
communication share should fall by about 4×. Measured at P = 8 it falls from
63.1 % to 10.7 %, a factor of 5.9 — the right direction and roughly the right
magnitude, falling somewhat *faster* than the pure surface/volume argument
predicts because the fixed per-collective latency (which does not grow with the
data at all) is amortised as well. **This is why the method is efficient only
when there is enough work to amortise one broadcast and one gather.**

### 8.3 Why P = 8 is slower than P = 4

The two effects combine. Between P = 4 and P = 8 at 1024³:

- compute barely moves (0.144480 → 0.143774 s) because there are no more
  physical cores to exploit;
- communication doubles (0.008497 → 0.016960 s) because there are twice as many
  ranks to feed and collect from.

Total therefore *rises*, 0.152106 → 0.157815 s. The turnover in Figure 1 is not
the algorithm failing — it is the point where added communication is no longer
paid for by added computation. Figure 4 makes this concrete: remove the hardware
constraint and P = 8 returns S = 5.97.

### 8.4 Cost of the Row-Row method specifically

Row-Row replicates `B` on every rank. That is its defining trade-off: it buys
**zero worker-to-worker communication** — each rank computes its rows of `C`
in complete isolation — at the price of one *n × p* broadcast and a memory
footprint of *n · p* integers per rank. The measurements show this is a good
bargain for large problems (at 1024³, P = 8 the broadcast is 6.3 % of runtime
and communication overall 10.7 %) and a poor one for small ones (at 256³, P = 8
the broadcast alone is 16.4 % and communication overall 63.1 %).

Among the three collectives, the scatter of `A` is the cheapest at every P > 1
(≤ 0.0018 s across all runs), because it is the only one whose per-rank volume
actually *falls* as P grows — each rank receives just *m/P × n* elements. The
gather of `C` costs consistently more than the scatter: `C` is accumulated in
64-bit for overflow safety, so it moves twice the bytes per element that `A`
does. The broadcast is the phase that grows fastest with P, which is the
structural price of replicating `B`.


## 9. Implementation notes

- **Compiler flags:** `-O2 -std=c++17`. MPI via HPC-X OpenMPI (`hpcx-2.7.0`).
- **Partition:** `debug`, single node (`node07`). Jobs 83598 (main) and 83608
  (physical-core control).
- **Non-divisible sizes:** handled with `MPI_Scatterv` / `MPI_Gatherv`; the first
  `m mod P` ranks take one extra row. Ranks with zero rows are valid and simply
  contribute nothing.
- **Launch:** `mpirun --oversubscribe` (P = 8 shares the node's cores) with
  `OMPI_MCA_coll_hcoll_enable=0` to suppress a benign hardware warning; both are
  set inside `run_q1.sh`, which is a submittable `sbatch` script.
- **Timing:** printed to stderr so it never contaminates the stdout result. The
  timed region spans broadcast → scatter → compute → gather (§6).
- **Count limits:** MPI collective element counts are `int`. The program checks
  every count and displacement against `INT_MAX` up front and aborts with a clear
  message rather than silently wrapping, which matters for the "1000 × 1000 or
  more" end of the input range.
- **Portability:** the sources use explicit standard headers rather than the
  GCC-only `<bits/stdc++.h>`, so they also build under clang.


## 10. Conclusion

The Row-Row MPI implementation is **correct** and **scales as the method
predicts**.

**Correctness.** 64/64 checks pass — 16 cases × P = 1, 2, 4, 8 — with output
compared byte-for-byte against the sequential reference. The suite covers both
worked examples from the assignment, *m* divisible and not divisible by *P*,
*m* = 1, *n* = 1, *p* = 1, tall and wide shapes, and *m* < *P* (idle ranks).

**Scaling.** The computation itself parallelises almost perfectly: compute-phase
speed-up is 3.97–3.99 out of an ideal 4.00 at P = 4 on *every* input shape,
including the skewed, non-divisible 1023 × 512 × 2048 case. End-to-end speed-up
reaches 3.80 at P = 4 for 1024³ (efficiency 0.95).

**The P = 8 turnover is hardware, and this was verified rather than assumed.**
The job held 8 logical CPUs = 4 physical cores, so the compute phase hit a hard
ceiling at ≈ 4.0× while communication kept growing with P. Re-running with
`--cpus-per-task=2` to obtain 8 genuine physical cores lifts S(8) from 3.66 to
**5.97** with no change to the source, confirming the limit is the CPU allocation
and not the decomposition.

**Communication vs computation.** Communication is a surface cost (*n·p* +
*m·n* + *m·p* elements) against a volume of work (*m·n·p* multiply-adds), so its
share of runtime falls sharply as the problem grows: at P = 8 it is 63.1 % of
runtime at 256³ but only 10.7 % at 1024³. The Row-Row method's defining
trade-off — replicate `B` everywhere to buy zero worker-to-worker communication —
is therefore a good bargain at scale and a poor one for small matrices.

---

# Graph Algorithms: Q6 — Connected Components of a Large Graph

## Results Table

Speed-up $S(P) = T_1 / T_P$, from the end-to-end runtime reported by the MPI
program.

| Input size | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|:-----:|:-----:|:-----:|:-----:|
| Small (V=5, E=3)          | 1.00 | 0.88 | 0.06 | 0.06 |
| Medium (V=500, E=2.5k)    | 1.00 | 0.83 | 0.07 | 0.06 |
| Large (V=5k, E=25k)       | 1.00 | 0.56 | 0.12 | 0.06 |
| Very large (V=100k, E=1M) | 1.00 | **0.75** | 0.46 | 0.42 |

## Runtime Table

End-to-end seconds: graph input/distribution + connected-component computation
+ output.

| Input size | Vertices | Undirected edges | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|---------:|-----------------:|------:|------:|------:|------:|
| Small      | 5       | 3         | 0.0007 | 0.0008 | 0.0126 | 0.0125 |
| Medium     | 500     | 2,500     | 0.0010 | 0.0012 | 0.0150 | 0.0165 |
| Large      | 5,000   | 25,000    | 0.0028 | 0.0050 | 0.0234 | 0.0469 |
| Very large | 100,000 | 1,000,000 | 0.0560 | 0.0750 | 0.1211 | 0.1331 |

The program's diagnostic output reports twice the undirected edge count, because
it counts adjacency-list entries and every generated edge occurs in both
endpoints' lists. Graphs generated by `gen_graph.py --suite` with seed 42; the
small case is the exact sample input from `task.md`.

## Efficiency

$E(P) = S(P)/P$, shown as a percentage.

| Input size | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|:-----:|:-----:|:-----:|:-----:|
| Small      | 100.00% | 43.75% | 1.39% | 0.70% |
| Medium     | 100.00% | 41.67% | 1.67% | 0.76% |
| Large      | 100.00% | 28.00% | 2.99% | 0.75% |
| Very large | 100.00% | 37.33% | 11.56% | 5.26% |

## Speed-up vs. $P$ graph

![Q6 speed-up](problem6/speedup.svg)

*Figure 6.1 — Measured speed-up for Q6. All curves stay below 1.*

## Communication vs. Computation

IPM measured the following percentage of aggregate process wall time inside MPI
calls. The remainder includes local computation, input, and output.

| Input size | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|:-----:|:-----:|:-----:|:-----:|
| Small      | 0.02% | 0.32% | 8.32% | 7.40% |
| Medium     | 0.02% | 0.31% | 10.71% | 12.46% |
| Large      | 0.03% | 4.23% | 15.30% | 26.93% |
| Very large | 0.77% | 18.27% | **45.80%** | **47.35%** |

For the very-large case at $P=8$, IPM attributes 17.12% of wall time to
`MPI_Scatterv`, 12.60% to `MPI_Barrier`, 9.29% to `MPI_Bcast`, and 8.33% to
`MPI_Allreduce`. Communication and synchronisation therefore consume almost half
of aggregate wall time at the largest process count.

**The workload is too small to parallelise profitably.** At the given maximum constraints ($V = 10^5$, $E = 10^6$) the entire component computation is only
~10 ms. Against that, one `Allreduce` of $10^5$ ints per round plus the scatter
of the edge set is not amortisable, and $S(P) < 1$ everywhere.

**The local computation does scale, once MPI time is subtracted.** The program
reports the MPI portion of its compute phase separately, so pure local work can
be isolated (very-large case):

| $P$ | compute phase | of which MPI | local compute only | local-only $S(P)$ |
|-----|:-------------:|:------------:|:------------------:|:-----------------:|
| 1 | 0.0117 | 0.0000 | 0.0117 | 1.00 |
| 2 | 0.0132 | 0.0008 | 0.0124 | 0.94 |
| 4 | 0.0222 | 0.0145 | 0.0077 | 1.52 |
| 8 | 0.0204 | 0.0152 | 0.0052 | **2.25** |



## Correctness

MPI output `diff`ed against `cc_seq` (sequential union-find) at every process
count.


## Implementation Notes

- **Compile flags:** `mpicxx -g -O2 -std=c++17` for the MPI program;
  `g++ -g -O2 -pg -std=c++17` for the sequential program (`-pg` for gprof).
  MPI via HPC-X OpenMPI 2.7.0.
- **Algorithm:** distributed label propagation accelerated by a local union-find.
  `lab[v]` starts at `v`; each round every process (1) seeds a union-find with
  what is already known — `union(i, lab[i])` for all `i`; (2) unions the
  endpoints of the edges of the vertices **it owns**; (3) sets
  `newlab[i] = find(i)` (a set's root is always its smallest id); (4)
  `MPI_Allreduce(MPI_MIN)` merges what every process learned. The loop stops when
  the label array stops changing. Labels only ever decrease and always name a
  vertex in the same true component, so termination is guaranteed; at the
  fixpoint labels are constant across every edge, which makes each label the
  component minimum. Because step 1 folds in *all* previously merged information,
  the merge reach roughly doubles per round — **2–4 rounds** in practice, even
  for a long chain (the worst case for plain one-hop propagation) split across 8
  processes.
- **Cost per round:** $O(V + E_{local}\cdot\alpha)$ compute, one `Allreduce` of
  $V$ ints.
- **Handling non-divisible sizes:** vertices are block-distributed and adjacency
  lists handed out with `MPI_Scatterv`, so no rank stores the whole edge set and
  a $V$ not divisible by $P$ needs no special case. The $V$-sized label array is
  replicated on every rank — 400 KB at $V \le 10^5$ — which is what makes the
  single-collective-per-round design worthwhile. Adjacency lists need not be
  symmetric; self-loops and duplicate edges are fine.
- **Launch:** `mpirun --bind-to none --oversubscribe`. `--bind-to none` is
  required because the default core binding refuses to launch when a node has
  fewer visible CPUs than ranks, which is the case at $P=8$ on this partition.
  `OMPI_MCA_coll_hcoll_enable=0` silences a benign "no HCA device" warning.
- **Profiling:** IPM via `LD_PRELOAD` with `IPM_REPORT=full` for the MPI
  breakdown; gprof for the sequential program.


## Conclusion

The implementation is **correct** but **does not speed up**: $S(P) < 1$ at every size and process count. The cause is
that the problem is latency-bound rather than compute-bound. At the assignment's
stated maximum the whole computation is ~10 ms, against which IPM measures 46–47 %
of wall time inside MPI calls at $P = 4$–8, and the multi-node allocation adds a
fixed ~10 ms of inter-node collective latency that alone exceeds the computation.
The local union-find work does parallelise — 2.25× at $P=8$ once MPI time is
subtracted — so the limitation is the communication pattern, not the local
algorithm. The targeted fix is boundary-only point-to-point exchange in place of
the full-array `Allreduce`, since boundary vertices grow with $P$ while each
rank's local share shrinks.


---

# Real-World Applications: Q8 — Large-Scale Weather & Environmental Data Analytics

## 1. Problem

Given `N` weather measurements — each a `timestamp station_id temperature
humidity pressure rainfall wind_speed` — compute a fixed set of statistics:
totals; average/min/max of temperature, humidity and pressure; total and max
rainfall; average and max wind speed; a count of extreme-temperature events
(temp ≥ 40 or ≤ 0); the hottest and coldest single measurements; the busiest
60-second interval; and the Top-K stations by measurement count. Both a
**sequential** and an **MPI** implementation are required, plus a reproducible
**dataset generator**.

The first input line is `N K S`, where `K` is the K of Top-K and `S` is the
number of stations (station ids `0..S-1`).

## 2. Approach

The statistics are almost all **reductions** (sums, min, max, counts), which
parallelise naturally: each process computes partial statistics over a slice of
the data, and the partials are combined.

The design centres on a single mergeable `Stats` object (in `weather_core.h`):

- `add(record)` folds one measurement into the running statistics.
- `merge_into(A, B)` combines two partial `Stats` so that the result equals
  processing all their records in one pass.

Because the sequential program, the MPI program, and the checker all use this
same `add`/`merge_into`, the parallel result is identical to the sequential one
by construction.

**MPI data flow:**
1. Rank 0 reads the file and **scatters** the `N` records across the `P` ranks
   (`MPI_Scatterv`, handling `N` not divisible by `P`).
2. Each rank builds a **local `Stats`** over its chunk.
3. The partial results are combined with **tree-based collective reductions**
   (§2.1) — not gathered to rank 0 for a serial merge.
4. Rank 0 formats and prints.

Per the assignment's freedom on input distribution, rank 0 reads and scatters;
what matters is that the *work* is distributed, which it is.

### 2.1 How the partials are combined

Every quantity in the output is a reduction, so each maps onto one MPI
collective:

| Quantity | Collective |
|---|---|
| `total`, `extreme` | `MPI_Reduce` `MPI_SUM` (2 × `long long`) |
| sums of temp/hum/pres/rain/wind | `MPI_Reduce` `MPI_SUM` (5 × `double`) |
| minima of temp/hum/pres | `MPI_Reduce` `MPI_MIN` (3 × `double`) |
| maxima of temp/hum/pres/rain/wind | `MPI_Reduce` `MPI_MAX` (5 × `double`) |
| per-station count / temp-sum / rain-sum | 3 × `MPI_Reduce` `MPI_SUM` of length `S` |
| 60-second interval histogram | `MPI_Allreduce` MIN/MAX for the range, then `MPI_Reduce` `MPI_SUM` of the dense array |
| hottest / coldest | `MPI_Gather` of `P` 24-byte structs, picked with the shared comparators |

A rank that receives zero records contributes reduction identities
(`+INFINITY` for minima, `-INFINITY` for maxima, an empty histogram), so
`P > N` is handled without a special case.


## 3. Implementation details

**Output formatting.** All floating-point outputs use exactly **6 decimal
places** (`%.6f`); counts, ids and timestamps are integers. Timing goes to
**stderr**, so stdout carries only the required result.

**Tie-breaking (implemented exactly as specified):**
- Top-K stations: count descending, then station id ascending.
- Hottest / coldest: extreme temperature; ties → smaller timestamp, then smaller
  station id.
- Busiest interval: `interval = timestamp / 60`; max count, ties → smaller
  interval id.

**Extreme events:** counted when `temperature ≥ 40.0` OR `temperature ≤ 0.0`.

**Edge cases:** `N = 1`, `S = 1`, `K` larger than the number of stations present
(Top-K then lists only the stations that appear), and empty categories are all
handled.

**Numerical note.** Floating-point addition is not associative, so a parallel
reduction can differ from a single pass in the last bit. In practice, across all
tests and process counts, the 6-decimal output was identical between sequential
and MPI (verified below); this is expected because the generated values have
limited precision and the sums stay well within double precision.

**The interval histogram: why it is a dense array, not a hash map.**
This is the single most important implementation decision in the program.

The busiest-interval statistic needs a count per 60-second bucket. The obvious
container is `unordered_map<interval_id, count>`, and that is what the first
version used. It is the wrong choice here, for a reason that only shows up at
scale: **the number of distinct buckets grows with `N`.** The supplied generator
spreads timestamps over `60·N/4` seconds, so a run with `N = 5 000 000` has about
**1.25 million distinct intervals**. That has two consequences:

1. *Computation.* Every record costs a hash lookup and a probable cache miss.
   Measured in isolation, building the histogram for 5M records takes **0.242 s**
   with `unordered_map` versus **0.007 s** with a dense array — a 34× difference,
   and it accounted for roughly half of the entire local-computation phase.
2. *Communication.* A hash map cannot be reduced by MPI. It had to be serialised
   (≈ 20 MB per rank at `N = 5M`), `MPI_Gatherv`-ed to rank 0, and merged there
   one rank at a time. So the combine phase moved **O(P·N)** bytes and did
   **O(P·N)** work on a single rank — it grew with `P` while the compute phase
   shrank. Measured, the merge step alone was **28–112× slower** than summing
   dense arrays.

The fix is to index a plain `std::vector<long long>` by `interval_id - base`
(`IntervalHist` in `weather_core.h`). It grows geometrically as records arrive,
so `add()` stays amortised O(1); and because it is one contiguous buffer, ranks
agree on a common range with two `MPI_Allreduce`s and then combine it with a
single `MPI_Reduce(MPI_SUM)` — a tree reduction of depth O(log P) in which rank 0
never receives more than one array.

The trade-off is memory: the array spans the full observed interval range, so a
dataset whose timestamps are extremely sparse (a handful of readings spread over
decades) would allocate far more buckets than it has records. `IntervalHist`
therefore refuses to allocate beyond `IVL_MAX_BUCKETS` (200 M buckets, 1.6 GB)
and exits with a clear message rather than thrashing. For the data this
assignment specifies — and for any realistically dense time series — the dense
representation is strictly better.


## 4. Compilation and execution

```bash
module load hpcx-2.7.0/hpcx-ompi
mpicxx -O2 -std=c++17 -o weather_mpi weather_mpi.cpp
g++    -O2 -std=c++17 -o weather_seq weather_seq.cpp
g++    -O2 -std=c++17 -o gen_weather gen_weather.cpp

mpirun -np 4 --oversubscribe ./weather_mpi input.txt        # print result
./weather_seq input.txt                                     # sequential
./gen_weather 1000000 10 1000 42 data_1M.txt                # generate dataset
```

## 5. Correctness verification

A 13-case suite (`tests/`) compares MPI output **byte-for-byte** against the
sequential reference for P = 1, 2, 4, 8. Cases cover the sample, `N=1`, `S=1`,
`K > #stations`, all-extreme temperatures, the busiest-interval tie, the
hottest/coldest temperature tie, many stations, a single station at scale, and
awkward sizes.

**Result:** `TOTAL: 52 passed, 0 failed — ALL TESTS PASSED`
(13 cases × 4 process counts). Full log: `tests/correctness_results.txt`.

Two further checks guard the parallel path specifically, both runnable without a
cluster:

- `weather_check.cpp` splits a dataset exactly as `MPI_Scatterv` would, merges
  the partials with `merge_into`, and asserts the result is **byte-identical** to
  a single sequential pass — the direct test that floating-point reduction order
  does not change the printed output.
- The same check was run against the rewritten collective combine path (scalar
  `MPI_SUM`/`MIN`/`MAX` reductions, per-station sums, interval-histogram rebase
  and sum, gathered hot/cold) for **P = 1, 2, 3, 4, 5, 7, 8, 16** on all 13
  cases — 104 combinations, all matching. The larger process counts deliberately
  exceed `N` on the small cases, exercising ranks that receive zero records.


## 6. Dataset generation

`gen_weather N K S seed [outfile]` produces reproducible datasets (mt19937,
fixed seed → identical file every time). Value ranges: temperature `[-10,45]`,
humidity `[0,100]`, pressure `[950,1050]`, rainfall `[0,50]`, wind `[0,150]`
(all to 1 decimal place); timestamp integer in `[0, 60·N/4]`; station id in
`[0, S-1]`. The benchmark datasets use seed 42, `S = 1000`, `K = 10`.


## 7. Benchmark methodology

- **Environment:** RCE SLURM cluster, `debug` partition, single node, OpenMPI
  (HPC-X 2.7.0), `mpicxx -O2 -std=c++17`.
- **CPU allocation.** `run_q8.sh` requests `--ntasks=8 --cpus-per-task=2` and
  launches with `mpirun --bind-to core`. The node has 2 SMT threads per physical
  core, so this gives each rank a **full physical core**. This matters: the
  earlier benchmark requested only `--ntasks=4`, so P = 4 and P = 8 were sharing
  4 hardware threads (2 physical cores) and no scaling measurement past P = 2 was
  meaningful. The script records `nproc`, `SLURM_CPUS_ON_NODE` and `lscpu` in its
  log so the allocation is documented rather than assumed.
- **Sizes:** `N = 100 000`, `1 000 000`, `5 000 000` (`S = 1000`, `K = 10`, seed 42).
- **Process counts:** P = 1, 2, 4, 8.
- **Both implementations are timed.** Each size is run once through
  `weather_seq` (sequential baseline) and then through `weather_mpi` at each P,
  so speed-up can be quoted against the sequential program and not only against
  the P = 1 MPI run.
- **Phase timing.** The timed region is split by barriers into three phases:
  `scatter` (distributing records — communication), `compute` (the local
  reduction — the parallelisable part), and `combine` (reducing the partials —
  communication). Because the phases are barrier-separated they sum exactly to
  the reported total.
- **Not timed:** rank 0's read and parse of the input file. This is deliberate —
  it is serial text I/O common to both implementations and would otherwise mask
  the parallel behaviour — but it is also, at these sizes, the largest single
  cost of actually running the program, and §9 returns to it.


## 8. Results

> **How to read this section.** §8.1–8.3 are the measurements from the *original*
> design (`unordered_map` histogram, gather-and-merge combine), kept deliberately
> as the "before" half of a before/after comparison — they are the evidence that
> motivated the redesign in §3. §8.4–8.6 are the rewritten implementation on the
> same hardware and datasets. Both halves come from `sbatch run_q8.sh`, which
> regenerates every table, plot and `results/benchmark_results.txt` from the raw
> log automatically.

### 8.1 Before the fix — total runtime (seconds)

Raw log: `results/bench_q8_before_fix.txt`.

| N | P=1 | P=2 | P=4 | P=8 |
|---|:---:|:---:|:---:|:---:|
| 100 000 | 0.007934 | 0.009098 | 0.018829 | 0.026428 |
| 1 000 000 | 0.114311 | 0.115914 | 0.190657 | 0.364176 |
| 5 000 000 | 1.037597 | 0.967898 | 1.284485 | 2.694831 |

### 8.2 Before the fix — speed-up  S(P) = T₁/T_P

| N | P=1 | P=2 | P=4 | P=8 |
|---|:---:|:---:|:---:|:---:|
| 100 000 | 1.00 | 0.87 | 0.42 | 0.30 |
| 1 000 000 | 1.00 | 0.99 | 0.60 | 0.31 |
| 5 000 000 | 1.00 | 1.07 | 0.81 | 0.39 |

**Every entry is at or below 1.00.** Adding processes made the program *slower* —
at P = 8 on the largest dataset, 2.6× slower than a single process.

### 8.3 Before the fix — phase breakdown (N = 5 000 000, seconds)

| P | scatter (comm) | compute | combine (comm) | total |
|---|:---:|:---:|:---:|:---:|
| 1 | 0.036745 | 0.520616 | 0.480236 | 1.037597 |
| 2 | 0.050670 | 0.300708 | 0.616520 | 0.967898 |
| 4 | 0.062562 | 0.243316 | 0.978607 | 1.284485 |
| 8 | 0.123588 | 0.341935 | 2.229307 | 2.694831 |

![Phase breakdown, before the fix](problem8/results/q8_phase_breakdown_before_fix.png)

*Figure 1 — Where the time went before the redesign. `compute` shrinks with P
while `combine` grows and swallows the run: 0.48 s → 2.23 s from P = 1 to P = 8.*

This table is the diagnosis. `combine` is not a fixed overhead — it **grows
roughly linearly with P**, because each rank serialised a ~20 MB interval map
and rank 0 merged all `P` of them one after another. Communication in the
combine phase carried O(P·N) bytes, so it could never be amortised by adding
processes; it was made worse by them.

### 8.4 After the fix — total runtime (seconds)

Raw log: `results/bench_q8.txt` (job 83690, 8 physical cores).

| N | P=1 | P=2 | P=4 | P=8 |
|---|:---:|:---:|:---:|:---:|
| 100 000 | 0.002060 | 0.003241 | 0.005262 | 0.007824 |
| 1 000 000 | 0.026538 | 0.025161 | 0.030416 | 0.033798 |
| 5 000 000 | 0.147561 | 0.144087 | 0.149491 | 0.149387 |

**Before vs after, same hardware, same datasets:**

| N | P=1 | P=2 | P=4 | P=8 |
|---|:---:|:---:|:---:|:---:|
| 100 000 | 3.9× | 2.8× | 3.6× | 3.4× |
| 1 000 000 | 4.3× | 4.6× | 6.3× | **10.8×** |
| 5 000 000 | 7.0× | 6.7× | 8.6× | **18.0×** |

*Speed-up of the redesigned program over the original at each point.* The
anti-scaling is gone: at N = 5M the P = 8 run went from **2.695 s to 0.149 s**,
and S(8) relative to P = 1 rose from 0.39 to 0.99. The combine phase alone fell
from 2.229 s to 0.062 s — a 36× reduction, matching the isolated prediction in §3.

### 8.5 Phase breakdown after the fix (N = 5 000 000, seconds)

| P | scatter (comm) | compute | combine (comm) | total | comm share |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.036195 | 0.097694 | 0.013673 | 0.147561 | 33.8 % |
| 2 | 0.039441 | 0.082471 | 0.022175 | 0.144087 | 42.8 % |
| 4 | 0.042763 | 0.058734 | 0.047993 | 0.149491 | 60.7 % |
| 8 | 0.045175 | 0.042279 | 0.061933 | 0.149387 | 71.7 % |

Compute-phase speed-up, the part that genuinely parallelises:

| N | P=1 | P=2 | P=4 | P=8 |
|---|:---:|:---:|:---:|:---:|
| 100 000 | 1.00 | 1.40 | 1.93 | 2.40 |
| 1 000 000 | 1.00 | 1.49 | 1.93 | 1.97 |
| 5 000 000 | 1.00 | 1.18 | 1.66 | 2.31 |

### 8.6 The result that matters: this workload is I/O-bound

The sequential program now times parsing and analytics **separately**, because
the MPI program's timed region excludes rank 0's file read. Timing the parse in
one and not the other would manufacture a speed-up that is really just "text
parsing is slow" — an earlier revision of this benchmark did exactly that and
appeared to show a 29× speed-up that did not exist.

With both measured honestly:

| N | serial text parse | analytics (sequential) | analytics as % of end-to-end |
|---|:---:|:---:|:---:|
| 100 000 | 0.084 s | 0.00126 s | **1.4 %** |
| 1 000 000 | 0.844 s | 0.01567 s | **1.7 %** |
| 5 000 000 | 4.219 s | 0.09860 s | **2.2 %** |

Parsing costs **29–45× more than the entire analytics**. And measured against
that honest sequential baseline, the MPI program never wins:

| N | P=1 | P=2 | P=4 | P=8 |
|---|:---:|:---:|:---:|:---:|
| 100 000 | 0.61 | 0.39 | 0.24 | 0.16 |
| 1 000 000 | 0.59 | 0.62 | 0.52 | 0.46 |
| 5 000 000 | 0.67 | 0.68 | 0.66 | 0.66 |

*Speed-up S(P) = T(sequential analytics) / T(MPI total).* Every value is below
1.00 — the MPI program's scatter and combine cost more than the computation they
distribute. This is not a defect being hidden; it is the honest and expected
result for a workload whose parallelisable part is 2 % of the job, and §9
explains precisely what would have to change.

![Total-time speed-up](problem8/results/q8_speedup.png)

*Figure 2 — Speed-up against both the MPI P = 1 run and the sequential analytics.*

![Compute-only speed-up](problem8/results/q8_compute_speedup.png)

*Figure 3 — The computation phase alone does scale, up to 2.4×.*

![Phase breakdown](problem8/results/q8_phase_breakdown.png)

*Figure 4 — After the redesign. `combine` no longer explodes; the run is now
limited by `scatter` and by communication overhead exceeding compute savings.*


## 9. Analysis: computation, communication, data distribution, scalability

**Computation.** The per-record work is a handful of floating-point updates plus
one histogram increment — embarrassingly parallel, no dependency between records.
The compute phase does scale, up to 2.4×. But note the per-record cost *rises*
with P (at N = 5M: 20 ns at P = 1, 33 ns at P = 2, 47 ns at P = 4, 68 ns at
P = 8). The cause is the dense histogram: its length is the observed *interval
range* (~N/4 ≈ 1.25 M buckets ≈ 10 MB at N = 5M), not the number of local
records, so every rank allocates and randomly walks a full-size array. Eight
ranks × 10 MB is an 80 MB working set against this node's 18 MB of L3, so the
phase becomes memory-bandwidth bound. That is the price of the dense
representation, and it is far cheaper than the hash map it replaced — but it is
the reason compute stops at ~2.4× rather than approaching 8×.

**Communication.** Two phases, behaving differently:

- `scatter` moves the raw records — O(N) bytes total, essentially fixed by the
  dataset. It grows only slowly with P (0.036 s → 0.045 s at N = 5M).
- `combine` moves the partial results. In the original design this was O(P·N)
  and dominated everything. With the dense histogram it is one `MPI_Reduce` over
  a fixed-length array, tree-reduced in O(log P): 2.229 s → 0.062 s at P = 8.

The general lesson, and the reason the original anti-scaled: **a reduction is
only cheap if the partial result is small.** Sums, minima, maxima and the
per-station arrays all reduce to something small and constant in N. The interval
histogram was the one piece of partial state that grew with the input, and it
alone dictated the program's scaling behaviour.

Even after the fix, communication is 34 %–72 % of the timed region at N = 5M and
up to 93 % at N = 100k. The computation being distributed is simply too small to
pay for distributing it.

**Data distribution — and the finding that dominates everything else.** Rank 0
reads and parses the whole file, then scatters it. Measuring that parse honestly
(§8.6) shows it is **29–45× the cost of the entire analytics**: at N = 5M,
4.219 s of parsing against 0.099 s of computation. The analytics is **2.2 % of
the end-to-end job.**

That single number explains every other result here. By Amdahl's law, a program
whose parallelisable portion is 2 % cannot be made meaningfully faster by
parallelising that portion — the ceiling is 1/(1−0.022) ≈ 1.02×, no matter how
many ranks are used. It is why the MPI program never beats the sequential
baseline (best 0.68×) despite the computation itself scaling 2.4×, and why the
redesign — which genuinely made the program 18× faster in absolute terms — did
not move the end-to-end picture at all.

It also names the three specific costs of the rank-0-reads design:

1. The read and parse are **serial**, and are the overwhelming majority of the work.
2. Rank 0 must hold the **entire dataset** — 56 bytes × N, or 280 MB at N = 5M —
   capping the problem size at one node's memory.
3. The scatter moves O(N) bytes regardless of P.

**Scalability, and what would actually fix it.** All three have the same remedy:
**parallel input**. If each rank opened the file and read only its own byte range
(snapping to line boundaries so no record is split), then the parse — the 98 % —
would itself parallelise, the scatter phase would disappear entirely, and no rank
would hold more than N/P records. On these measurements that would take the
N = 5M end-to-end time from ≈ 4.32 s to roughly 4.22/8 + 0.10 ≈ 0.63 s, an
end-to-end speed-up near 6.5× — the difference between a program that
demonstrates parallelism and one that benefits from it.

This is the correct next step and it is deliberately named rather than
implemented: it is a structural change to how the program ingests data, and the
work in this report was scoped to making the *analytics* correct and scalable.
The measurements above are what identify it as the right change, and that
identification is itself the main analytical result of this question.


## 10. Implementation notes

- **Compiler:** `-O2 -std=c++17`, OpenMPI HPC-X 2.7.0 on the `debug` partition.
- **Distribution:** `MPI_Scatterv` (records as raw bytes) handles `N` not
  divisible by `P`; ranks with zero records are valid.
- **Combine:** tree-based `MPI_Reduce` / `MPI_Allreduce` collectives (§2.1). The
  sequential program and the checker still use `merge_into`, so the tie-break and
  aggregation rules live in exactly one place (`weather_core.h`) and cannot drift
  between the two implementations.
- **Count limits:** MPI counts are `int`. The scatter moves `N × sizeof(Rec)`
  = 56 N bytes, which overflows `int` at N ≈ 38.3 M; the program checks this up
  front and aborts with a clear message instead of silently corrupting the
  transfer. The interval reduction is guarded the same way.
- **Launch:** `mpirun --bind-to core` with `--cpus-per-task=2` so each rank owns
  a physical core, and `OMPI_MCA_coll_hcoll_enable=0` to silence a benign HCA
  warning. Both are set inside `run_q8.sh`, a submittable `sbatch` script.
- **Portability:** explicit standard headers rather than the GCC-only
  `<bits/stdc++.h>`, so the sources also build under clang.
- **Output:** floats to 6 decimals; timing to stderr only, so stdout carries
  exactly the required result.


## 11. Conclusion

**Correctness.** 52/52 cluster checks (13 cases × P = 1, 2, 4, 8) plus 104
host-side checks of the collective combine path across P = 1…16, every one
byte-identical to the sequential reference. Output format, field order and all
three tie-break rules match the specification exactly; the tiny sample was
verified by hand.

**The engineering result.** The first version *anti-scaled* — P = 8 was 2.6×
slower than P = 1 — and the phase breakdown localised the cause to the combine
phase. The reason was a data-structure choice, not anything about MPI: holding
the 60-second interval histogram in an `unordered_map` meant the partial state
grew with N, so combining it moved O(P·N) bytes into a serial merge on rank 0.
Replacing it with a dense array made the same statistic a contiguous buffer that
`MPI_Reduce(MPI_SUM)` combines in O(log P). Measured on the cluster: the P = 8
run at N = 5M went from **2.695 s to 0.149 s (18×)**, the combine phase from
2.229 s to 0.062 s (36×), and S(8) from 0.39 to 0.99.

**The honest result.** That 18× improvement does not translate into beating the
sequential program, and the corrected benchmark shows exactly why. Parsing the
input costs 29–45× more than the entire analytics, so the parallelisable part of
this job is about **2 % of it**. Amdahl's law caps the achievable end-to-end
gain at ≈ 1.02× regardless of process count, and the measured speed-up against a
like-for-like sequential baseline is 0.66–0.68×: the scatter and combine cost
more than the computation they distribute. An earlier revision of this benchmark
timed parsing in the sequential program but not in the MPI one and appeared to
show a 29× speed-up; that number was an artefact of mismatched timers and is not
reported here.

**The lessons.** Two, and they are the substance of this question. First: in a
reduction-shaped workload, scalability is decided not by how much data you start
with but by **how much partial state each rank hands back** — one quantity whose
partial state grew with the input dictated the behaviour of the entire program.
Second: **profile the whole job before parallelising part of it.** The analytics
here was made 18× faster and it changed nothing end-to-end, because the 98 % was
somewhere else entirely. Parallel input (each rank reading its own byte range) is
the change that would matter, and it is identified in §9 with the measurements
that justify it.
