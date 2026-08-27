module load hpcx-2.7.0/hpcx-ompi 2>/dev/null

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST | tasks=$SLURM_NTASKS"

# Compile 
mpicxx -O2 -std=c++17 -o rowrow rowrow_matmul.cpp || { echo "MPI compile failed"; exit 1; }
g++    -O2 -std=c++17 -o seq    seq_matmul.cpp     || { echo "seq compile failed"; exit 1; }
g++    -O2 -std=c++17 -o gen    gen_matrix.cpp     || { echo "gen compile failed"; exit 1; }

# Correctness: generate a small uneven case, compare MPI vs sequential
./gen 7 5 4 1 case_small.txt        # 7 rows -> not divisible by 2 or 4
./seq case_small.txt seq_out.txt
echo "--- Correctness (should all say PASS) ---"
for P in 1 2 4 8; do
    mpirun -np $P ./rowrow case_small.txt mpi_out_$P.txt 2>/dev/null
    if diff -q seq_out.txt mpi_out_$P.txt >/dev/null; then echo "P=$P  PASS"; else echo "P=$P  FAIL"; fi
done

# Benchmark: square sizes x process counts. Timing lines go to stderr (the .err file). 
echo "--- Timing (see the .err file for 'time=' lines) ---"
for SZ in 256 512 1024; do
    for P in 1 2 4 8; do
        mpirun -np $P ./rowrow --gen $SZ $SZ $SZ 42
    done
done

echo "Done."
