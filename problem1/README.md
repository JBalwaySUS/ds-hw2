# Q1 — Distributed Matrix Multiplication (Row-Row Method, MPI)

Computes `C = A * B` where `A` is `m x n` and `B` is `n x p`.
`A` is split **row-wise** across processes; `B` is **broadcast** in full; each
process computes its own rows of `C` (a weighted sum of the rows of `B`) with no
worker-to-worker communication. The master gathers the row-slices back together.

## Files
- `rowrow_matmul.cpp` — MPI implementation.
- `seq_matmul.cpp` — sequential reference (for correctness checking).
- `gen_matrix.cpp` — reproducible input generator.
- `run_q1.sh` — SLURM batch script (compile + correctness + benchmark + plots).
- `plot_speedup.py` — builds the speed-up / efficiency / phase-breakdown plots.
- `tests/` — 16-case correctness suite, run against P = 1, 2, 4, 8.
- `results/` — raw timings, benchmark table, plots, SLURM job log.

## Input format
```
m n p
<m*n integers : A in row-major order>
<n*p integers : B in row-major order>
```
Output: `m` lines of `p` space-separated integers (matrix C) on stdout.
Timing is printed to **stderr**, so stdout stays clean. The stderr line carries
the total parallel-phase time plus a per-phase breakdown:

```
P=4  m=1024 n=1024 p=1024  time=0.291764 s  bcastB=... scatterA=... compute=... gatherC=...
```

## Compile
```
mpicxx -O2 -std=c++17 -o rowrow rowrow_matmul.cpp
g++    -O2 -std=c++17 -o seq    seq_matmul.cpp
g++    -O2 -std=c++17 -o gen    gen_matrix.cpp
```

## Run
```
mpirun -np 4 ./rowrow input.txt              # print C to stdout
mpirun -np 4 ./rowrow input.txt out.txt      # write C to out.txt
mpirun -np 4 ./rowrow --gen 1000 1000 1000 42 # generate internally; time only
```
On SLURM you may use `srun ./rowrow input.txt` instead of `mpirun -np ...`
(the process count then comes from the allocation).

### Everything at once
```
sbatch run_q1.sh        # from inside Q1/ : compile, verify, benchmark, plot
```
This writes `results/bench.txt`, refreshes `tests/correctness_results.txt`, and
regenerates `results/{speedup,efficiency,phase_breakdown}.png`. It requests the
`debug` partition with 8 tasks and launches with `--oversubscribe` plus
`OMPI_MCA_coll_hcoll_enable=0`, and logs `nproc` / `lscpu` so the scaling
analysis can be checked against the node's real core count.

## Correctness check
```
./gen 7 5 4 1 case.txt          # 7 rows: deliberately not divisible by 2/4
./seq case.txt seq_out.txt
mpirun -np 4 ./rowrow case.txt mpi_out.txt
diff seq_out.txt mpi_out.txt    # no output = identical = correct
```
Also verify the two textbook examples give:
- `3 2 3` example -> `4 3 2 / 3 0 -3 / 2 -3 -8`
- `4 2 2` example -> `1 2 / 2 3 / 0 3 / 1 3`

## Handling of required cases
- **m not divisible by P**: `MPI_Scatterv`/`MPI_Gatherv` with per-rank counts;
  the first `m % P` ranks each take one extra row.
- **Edge cases** m=1, n=1, p=1, tall (m>>n), wide (n>>m): all handled; a rank
  that receives 0 rows simply does nothing.
- **Overflow**: products accumulate in 64-bit (`long long`). MPI element counts
  are `int`, so every count and displacement is checked against `INT_MAX` at
  startup and the program aborts with a clear message rather than wrapping.

## Note on the generator
`gen_matrix.cpp` is reproducible for a fixed seed *on a given standard library*.
`std::uniform_int_distribution` is implementation-defined, so libstdc++ (the
cluster) and libc++ (macOS) produce different streams for the same seed. The
`tests/*.in` and `tests/*.out` pairs are committed together, so this does not
affect correctness checking.