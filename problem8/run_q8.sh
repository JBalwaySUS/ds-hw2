#!/bin/bash
#SBATCH --job-name=q8_weather
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --output=results/q8_%j.out
#SBATCH --error=results/q8_%j.err
#
# Q8 - Weather analytics: compile, verify correctness, benchmark BOTH the
# sequential and the MPI implementation, then plot.
# Submit with:  sbatch run_q8.sh      (from inside the Q8 directory)
#
# --cpus-per-task=2 gives each rank a full physical core (the node has 2 SMT
# threads per core), so P=8 runs on 8 real cores rather than 4 cores' worth of
# hyperthreads. Without it, scaling measurements past P=4 are meaningless.

set -u
mkdir -p results
module load hpcx-2.7.0/hpcx-ompi 2>/dev/null
export OMPI_MCA_coll_hcoll_enable=0
MPIRUN="mpirun --bind-to core"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST | tasks=$SLURM_NTASKS"
echo "--- Node hardware ---"
echo "nproc                  : $(nproc)"
echo "SLURM_CPUS_ON_NODE     : ${SLURM_CPUS_ON_NODE:-unset}"
echo "SLURM_CPUS_PER_TASK    : ${SLURM_CPUS_PER_TASK:-unset}"
lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Core\(s\) per socket|Socket\(s\)|Thread\(s\) per core)' || true
echo

# --- compile ---
mpicxx -O2 -std=c++17 -o weather_mpi   weather_mpi.cpp   || { echo "MPI compile failed"; exit 1; }
g++    -O2 -std=c++17 -o weather_seq   weather_seq.cpp   || { echo "seq compile failed"; exit 1; }
g++    -O2 -std=c++17 -o gen_weather   gen_weather.cpp   || { echo "gen compile failed"; exit 1; }
g++    -O2 -std=c++17 -o weather_check weather_check.cpp || { echo "check compile failed"; exit 1; }

# --- correctness: MPI vs sequential on a generated case, P=1,2,4,8 ---
./gen_weather 5000 5 50 42 case.txt
./weather_seq case.txt seq_out.txt 2>/dev/null
echo "--- Correctness (want all PASS) ---"
for P in 1 2 4 8; do
    $MPIRUN -np $P ./weather_mpi case.txt mpi_$P.txt 2>/dev/null
    if diff -q seq_out.txt mpi_$P.txt >/dev/null; then echo "P=$P PASS"; else echo "P=$P FAIL"; fi
done

# --- full 13-case suite across P = 1, 2, 4, 8 ---
echo "--- Full test suite ---"
( cd tests && bash run_tests.sh ../weather_mpi ) | tee tests/correctness_results.txt

# --- benchmark: BOTH implementations, multiple sizes x process counts ---
echo "--- Benchmark (raw lines also land in results/bench_q8.txt) ---"
: > results/bench_q8.txt
for N in 100000 1000000 5000000; do
    ./gen_weather $N 10 1000 42 data_$N.txt
    # sequential baseline (the assignment asks for both implementations timed)
    ./weather_seq data_$N.txt /dev/null 2>> results/bench_q8.txt
    for P in 1 2 4 8; do
        $MPIRUN -np $P ./weather_mpi data_$N.txt /dev/null 2>> results/bench_q8.txt
    done
    rm -f data_$N.txt      # datasets are regenerable from the fixed seed
done
cat results/bench_q8.txt

# --- plots ---
python3 plot_q8.py results/bench_q8.txt || echo "plotting skipped (matplotlib unavailable)"

echo "Done."
