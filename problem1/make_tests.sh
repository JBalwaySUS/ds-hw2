#!/bin/bash
set -e
g++ -O2 -std=c++17 -o gen gen_matrix.cpp
g++ -O2 -std=c++17 -o seq seq_matmul.cpp
mkdir -p tests && cd tests
cat > 01_pdf_example1.in <<'EOF'
3 2 3
1 2
0 3
-1 4
2 3 4
1 0 -1
EOF
cat > 02_pdf_example2.in <<'EOF'
4 2 2
1 0
2 -1
0 3
1 1
1 2
0 1
EOF
cat > 03_one_by_one.in <<'EOF'
1 1 1
7
6
EOF
cat > 04_identity.in <<'EOF'
4 4 4
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
EOF
cat > 05_zeros.in <<'EOF'
4 4 3
0 0 0 0
1 0 2 0
0 0 0 0
0 3 0 4
1 1 1
2 2 2
3 3 3
4 4 4
EOF
../gen 8  4  5  1  06_even_split_8x4x5.in
../gen 7  5  4  2  07_uneven_split_7x5x4.in
../gen 1  6  5  3  08_single_row_1x6x5.in
../gen 5  1  4  4  09_n_equals_1_5x1x4.in
../gen 5  4  1  5  10_p_equals_1_5x4x1.in
../gen 30 3  4  6  11_tall_30x3x4.in
../gen 3  30 4  7  12_wide_3x30x4.in
../gen 3  5  5  8  13_fewer_rows_than_P_3x5x5.in
../gen 13 11 7  9  14_odd_dims_13x11x7.in
../gen 10 10 10 10 15_square_10x10x10.in
../gen 100 80 60 11 16_medium_100x80x60.in
for f in *.in; do ../seq "$f" "${f%.in}.out"; done
cat > run_tests.sh <<'EOF'
#!/bin/bash
module load hpcx-2.7.0/hpcx-ompi 2>/dev/null
export OMPI_MCA_coll_hcoll_enable=0
MPIRUN="mpirun --oversubscribe"
BIN=${1:-../rowrow}
if [ ! -x "$BIN" ]; then echo "MPI binary not found at '$BIN'. Compile rowrow first."; exit 1; fi
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
EOF
cd ..
echo "Created tests/ with $(ls tests/*.in | wc -l) cases."
