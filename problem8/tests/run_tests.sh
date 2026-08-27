#!/bin/bash
module load hpcx-2.7.0/hpcx-ompi 2>/dev/null
export OMPI_MCA_coll_hcoll_enable=0
MPIRUN="mpirun --oversubscribe"
BIN=${1:-../weather_mpi}
if [ ! -x "$BIN" ]; then echo "MPI binary not found at '$BIN'. Compile weather_mpi first."; exit 1; fi
tmp=$(mktemp -d); pass=0; fail=0; failed=""
for infile in *.in; do
    base=${infile%.in}; line="$base"
    for P in 1 2 4 8; do
        $MPIRUN -np $P "$BIN" "$infile" "$tmp/out" 2>/dev/null
        if diff -q "$base.out" "$tmp/out" >/dev/null 2>&1; then pass=$((pass+1)); line="$line  P$P:PASS"
        else fail=$((fail+1)); line="$line  P$P:FAIL"; failed="$failed $base(P=$P)"; fi
    done
    echo "$line"
done
rm -rf "$tmp"
echo "------------------------------------------------------------"
echo "TOTAL: $pass passed, $fail failed"
[ $fail -gt 0 ] && echo "Failing:$failed" || echo "ALL TESTS PASSED"
