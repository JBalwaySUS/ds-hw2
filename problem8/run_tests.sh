#!/bin/bash
# run_tests.sh - run every *.in test through the MPI program for P=1,2,4,8 and
# compare its output to the pre-computed *.out (correct answer from the
# sequential program). A test PASSES only if all four process counts match.
#
# USAGE (inside a compute-node allocation, from within this tests/ folder):
#   bash run_tests.sh                    # assumes ../weather_mpi
#   bash run_tests.sh /path/to/weather_mpi

module load hpcx-2.7.0/hpcx-ompi 2>/dev/null
export OMPI_MCA_coll_hcoll_enable=0
MPIRUN="mpirun --oversubscribe"

BIN=${1:-../weather_mpi}
if [ ! -x "$BIN" ]; then
    echo "ERROR: MPI binary not found at '$BIN'. Compile it first:"
    echo "  mpicxx -O2 -std=c++17 -o weather_mpi ../weather_mpi.cpp"
    exit 1
fi

tmp=$(mktemp -d); pass=0; fail=0; failed=""
for infile in *.in; do
    [ -e "$infile" ] || continue
    base=${infile%.in}; line="$base"
    for P in 1 2 4 8; do
        $MPIRUN -np $P "$BIN" "$infile" "$tmp/out" 2>/dev/null
        if diff -q "$base.out" "$tmp/out" >/dev/null 2>&1; then
            pass=$((pass+1)); line="$line  P$P:PASS"
        else
            fail=$((fail+1)); line="$line  P$P:FAIL"; failed="$failed $base(P=$P)"
        fi
    done
    echo "$line"
done
rm -rf "$tmp"
echo "------------------------------------------------------------"
echo "TOTAL: $pass passed, $fail failed"
[ $fail -gt 0 ] && echo "Failing:$failed" || echo "ALL TESTS PASSED"
