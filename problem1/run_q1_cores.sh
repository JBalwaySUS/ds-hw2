#!/bin/bash
#SBATCH --job-name=q1_cores
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=results/q1_cores_%j.out
#SBATCH --error=results/q1_cores_%j.err
#
# Control experiment for the P=8 plateau in the main benchmark.
# --cpus-per-task=2 asks for 2 hardware threads per rank, i.e. one FULL
# physical core per rank instead of a single SMT sibling.
#   - If P=8 now scales, the plateau was hyperthreading.
#   - If it still flattens, the limit is memory bandwidth instead.

set -u
mkdir -p results
module load hpcx-2.7.0/hpcx-ompi 2>/dev/null
export OMPI_MCA_coll_hcoll_enable=0

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "nproc               : $(nproc)"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo

mpicxx -O2 -std=c++17 -o rowrow rowrow_matmul.cpp || exit 1

: > results/bench_physical_cores.txt
for DIMS in "1024 1024 1024" "1023 512 2048"; do
    for P in 1 2 4 8; do
        mpirun --bind-to core -np $P ./rowrow --gen $DIMS 42 2>> results/bench_physical_cores.txt
    done
done
cat results/bench_physical_cores.txt
echo "Done."
