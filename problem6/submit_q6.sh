#!/bin/bash
#SBATCH --job-name=mpi-connected-components
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=q6_%j.log
#SBATCH --error=q6_%j.err
#SBATCH --partition=debug
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@college.edu

# Load necessary modules. HPC-X is the module that works multi-node on this
# cluster (openmpi/4.1.5 has no working UCX and its TCP fallback picks the
# wrong interface, so runs spanning nodes hang). IPM, the MPI profiler, ships
# inside HPC-X at $HPCX_IPM_LIB and is enabled by LD_PRELOAD - nothing to build.
module load hpcx-2.7.0/hpcx-ompi

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "========================================="
echo ""

# Compile with profiling enabled: IPM for the MPI program, gprof for the sequential one
echo "Compiling MPI program..."
mpicxx -g -O2 -std=c++17 -o cc_mpi cc_mpi.cpp
if [ $? -ne 0 ]; then
    echo "MPI compilation failed!"
    exit 1
fi

echo "Compiling sequential program (with -pg)..."
g++ -g -O2 -pg -std=c++17 -o cc_seq cc_seq.cpp
if [ $? -ne 0 ]; then
    echo "Sequential compilation failed!"
    exit 1
fi

# hcoll cannot find an HCA device here and prints a wall of warnings; disable it
export OMPI_MCA_coll_hcoll_enable=0

# --bind-to none: the default core binding refuses to launch when a node has
# fewer visible cpus than ranks, which is the case at P=8 on this partition
MPIRUN="mpirun --bind-to none --oversubscribe"
IPM="-x LD_PRELOAD=$HPCX_IPM_LIB -x IPM_REPORT=full"

# Generate test graphs
echo ""
echo "Generating test graphs..."
python3 generate.py --suite -d data

echo ""
echo "Starting benchmark tests..."
echo ""

# Run each input size at P = 1, 2, 4, 8; verify against the sequential run each time
for f in data/*.txt; do
    echo "========================================="
    echo "Input: $f"
    echo "========================================="

    rm -f seq_out.txt
    ./cc_seq $f seq_out.txt 2>&1

    for P in 1 2 4 8; do
        echo ""
        echo "--- P=$P ---"
        # delete the previous output first, so a failed launch cannot "pass"
        # by being diffed against the file the last run left behind
        rm -f mpi_out.txt
        $MPIRUN -np $P $IPM ./cc_mpi $f mpi_out.txt 2>&1
        rc=$?

        if [ $rc -ne 0 ]; then
            echo "Correctness: FAIL (mpirun exited $rc)"
        elif [ ! -s mpi_out.txt ]; then
            echo "Correctness: FAIL (no output produced)"
        elif diff -q seq_out.txt mpi_out.txt > /dev/null; then
            echo "Correctness: PASS"
        else
            echo "Correctness: FAIL (output differs from sequential)"
        fi
    done
    echo ""
done

# gprof profile of the sequential program (from the last ./cc_seq run)
echo "========================================="
echo "gprof profile of sequential program"
echo "========================================="
gprof ./cc_seq gmon.out 2>&1 | head -40

echo ""
echo "========================================="
echo "Benchmark completed!"
echo "Results saved in: q6_$SLURM_JOB_ID.log"
echo "========================================="
