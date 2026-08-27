# Q8 — Large-Scale Weather & Environmental Data Analytics (MPI)

Computes a fixed set of statistics over `N` weather measurements: totals,
averages, min/max of temperature/humidity/pressure, rainfall and wind stats,
extreme-temperature event counts, hottest/coldest measurements, the busiest
60-second interval, and the Top-K stations by measurement count.

## Files
- `weather_mpi.cpp` — MPI implementation (the parallel program).
- `weather_seq.cpp` — sequential implementation (reference / correctness ground truth).
- `gen_weather.cpp` — reproducible dataset generator (fixed seed).
- `weather_core.h` — shared logic (record type, statistics accumulator, merge,
  output formatting) used by all of the above so results are identical.
- `weather_check.cpp` — internal validator (split-merge vs single pass).
- `run_q8.sh` — SLURM batch script (compile + correctness + benchmark).
- `plot_q8.py` — builds the speed-up / phase-breakdown plots from the timing log.
- `tests/` — 13-case correctness suite + `correctness_results.txt`.

## Input format
```
N K S
<N lines: timestamp station_id temperature humidity pressure rainfall wind_speed>
```
- `N` = number of measurements, `K` = K for Top-K, `S` = number of stations
  (station_id in `[0, S-1]`).
- `timestamp` and `station_id` are integers; the five weather values are floats.

## Output format
Exactly the fields listed in the assignment, in order. All floating-point values
are printed with **6 digits after the decimal point**. Counts, IDs and timestamps
are printed as integers. Timing information is written to **stderr**, never stdout.

Tie-breaks: Top-K stations by count desc then station id asc; hottest/coldest by
temperature, ties broken by smaller timestamp then smaller station id; busiest
interval by count, ties broken by smaller interval id (`interval = timestamp/60`).

## Compile
```bash
module load hpcx-2.7.0/hpcx-ompi        # on the RCE cluster
mpicxx -O2 -std=c++17 -o weather_mpi   weather_mpi.cpp
g++    -O2 -std=c++17 -o weather_seq   weather_seq.cpp
g++    -O2 -std=c++17 -o gen_weather   gen_weather.cpp
g++    -O2 -std=c++17 -o weather_check weather_check.cpp
```

## Run
```bash
mpirun -np 4 --oversubscribe ./weather_mpi input.txt           # print to stdout
mpirun -np 4 --oversubscribe ./weather_mpi input.txt out.txt   # write to out.txt
./weather_seq input.txt                                        # sequential
```
On the RCE cluster, launch from inside a compute-node allocation and set
`export OMPI_MCA_coll_hcoll_enable=0` to silence a benign HCA warning.

## Generate a reproducible dataset
```bash
./gen_weather N K S seed [outfile]
# e.g.
./gen_weather 1000000 10 1000 42 data_1M.txt
```
Value ranges (documented for reproducibility), floats to 1 decimal place:
temperature `[-10,45]`, humidity `[0,100]`, pressure `[950,1050]`,
rainfall `[0,50]`, wind `[0,150]`; timestamp integer in `[0, 60*N/4]`;
station_id integer in `[0, S-1]`. Same `(N,K,S,seed)` always yields the same file.

## Correctness check
```bash
./gen_weather 5000 5 50 42 case.txt
./weather_seq case.txt seq_out.txt
mpirun -np 4 --oversubscribe ./weather_mpi case.txt mpi_out.txt
diff seq_out.txt mpi_out.txt        # no output = identical = correct
```
Or run the full suite:
```bash
cd tests && bash run_tests.sh       # expects "ALL TESTS PASSED" (52/52)
```

## Benchmark
```bash
sbatch run_q8.sh                    # or run the sweep manually inside an allocation
```
The MPI program prints a timing line to stderr with a phase breakdown:
```
MPI  P=4 N=1000000 K=10 S=1000  time=... s  scatter=... compute=... combine=...
```
`scatter` = data distribution, `compute` = local computation, `combine` =
gather + merge on rank 0. Feed the captured lines to `plot_q8.py` for the plots.

## Parallelisation strategy
Rank 0 reads the file and scatters the records across ranks (`MPI_Scatterv`,
handles `N` not divisible by `P`). Each rank computes a local statistics object
over its chunk; the partials are gathered to rank 0 and merged with the same
`merge_into` used everywhere, then rank 0 formats and prints. This makes the MPI
result identical to the sequential one (verified: 52/52 across P=1,2,4,8).
