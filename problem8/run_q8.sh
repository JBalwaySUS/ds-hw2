#!/bin/bash
#SBATCH --job-name=weather-mpi
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=weather_%j.log
#SBATCH --error=weather_%j.err
#SBATCH --partition=debug

# --- environment (fixes learned on this cluster) ---
module load hpcx-2.7.0/hpcx-ompi 2>/dev/null
export OMPI_MCA_coll_hcoll_enable=0
MPIRUN="mpirun --oversubscribe"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST | tasks=$SLURM_NTASKS"

# --- compile ---
mpicxx -O2 -std=c++17 -o weather_mpi   weather_mpi.cpp   || { echo "MPI compile failed"; exit 1; }
g++    -O2 -std=c++17 -o weather_seq   weather_seq.cpp   || { echo "seq compile failed"; exit 1; }
g++    -O2 -std=c++17 -o gen_weather   gen_weather.cpp   || { echo "gen compile failed"; exit 1; }

# --- correctness: MPI vs sequential on a generated case, P=1,2,4,8 ---
./gen_weather 5000 5 50 42 case.txt
./weather_seq case.txt seq_out.txt
echo "--- Correctness (want all PASS) ---"
for P in 1 2 4 8; do
    $MPIRUN -np $P ./weather_mpi case.txt mpi_$P.txt
    if diff -q seq_out.txt mpi_$P.txt >/dev/null; then echo "P=$P PASS"; else echo "P=$P FAIL"; fi
done

# --- benchmark: sizes x process counts. Phase-timed lines go to the .err file. ---
echo "--- Benchmark (timings in the .err file) ---"
for N in 100000 1000000 5000000; do
    ./gen_weather $N 10 1000 42 data_$N.txt
    for P in 1 2 4 8; do
        $MPIRUN -np $P ./weather_mpi data_$N.txt /dev/null
    done
    rm -f data_$N.txt      # datasets are regenerable; don't keep the big files
done

echo "Done."
