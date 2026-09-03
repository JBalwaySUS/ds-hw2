#!/bin/bash
#SBATCH --job-name=q1_rowrow
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:20:00
#SBATCH --output=results/q1_%j.out
#SBATCH --error=results/q1_%j.err
#
# Q1 - Row-Row matrix multiplication: compile, verify correctness, benchmark.
# Submit with:  sbatch run_q1.sh      (from inside the Q1 directory)

set -u
mkdir -p results

module load hpcx-2.7.0/hpcx-ompi 2>/dev/null

# The debug node exposes fewer cores than 8, so P=8 must share cores.
# hcoll is disabled to silence a benign HCA hardware warning.
export OMPI_MCA_coll_hcoll_enable=0
MPIRUN="mpirun --oversubscribe"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST | tasks=$SLURM_NTASKS"

# Record the hardware the scaling results must be interpreted against.
echo "--- Node hardware ---"
echo "nproc                 : $(nproc)"
echo "SLURM_CPUS_ON_NODE    : ${SLURM_CPUS_ON_NODE:-unset}"
echo "SLURM_JOB_CPUS_PER_NODE: ${SLURM_JOB_CPUS_PER_NODE:-unset}"
lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Core\(s\) per socket|Socket\(s\)|Thread\(s\) per core)' || true
echo

# Compile
mpicxx -O2 -std=c++17 -o rowrow rowrow_matmul.cpp || { echo "MPI compile failed"; exit 1; }
g++    -O2 -std=c++17 -o seq    seq_matmul.cpp     || { echo "seq compile failed"; exit 1; }
g++    -O2 -std=c++17 -o gen    gen_matrix.cpp     || { echo "gen compile failed"; exit 1; }

# Correctness: small uneven case (7 rows -> not divisible by 2 or 4), MPI vs sequential
./gen 7 5 4 1 case.txt
./seq case.txt seq_out.txt
echo "--- Correctness (should all say PASS) ---"
for P in 1 2 4 8; do
    $MPIRUN -np $P ./rowrow case.txt mpi_out_$P.txt 2>/dev/null
    if diff -q seq_out.txt mpi_out_$P.txt >/dev/null; then echo "P=$P  PASS"; else echo "P=$P  FAIL"; fi
done

# Full 16-case suite across P = 1, 2, 4, 8
echo "--- Full test suite ---"
( cd tests && bash run_tests.sh ../rowrow ) | tee tests/correctness_results.txt

# Benchmark: square sizes x process counts, plus one skewed, non-P-divisible case.
# Timing lines go to stderr; capture them so the plotting script can read them.
echo "--- Timing (raw lines also land in results/bench.txt) ---"
: > results/bench.txt
for DIMS in "256 256 256" "512 512 512" "1024 1024 1024" "1023 512 2048"; do
    for P in 1 2 4 8; do
        $MPIRUN -np $P ./rowrow --gen $DIMS 42 2>> results/bench.txt
    done
done
cat results/bench.txt

# Plots (written into results/)
python3 plot_speedup.py results/bench.txt || echo "plotting skipped (matplotlib unavailable)"

echo "Done."
