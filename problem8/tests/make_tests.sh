#!/bin/bash
# make_tests.sh - regenerate the Q8 test suite from scratch (no upload needed).
# Run from inside your Q8 folder. Rebuilds gen_weather/weather_seq, creates tests/,
# writes all inputs, and computes the expected answers.
set -e
g++ -O2 -std=c++17 -o gen_weather gen_weather.cpp
g++ -O2 -std=c++17 -o weather_seq weather_seq.cpp
mkdir -p tests && cd tests

cat > 01_tiny_sample.in <<'EOF'
5 2 3
0 0 40.0 50 1000 0 10
30 1 -5.0 60 1010 5 20
70 0 25.0 55 1005 2 15
90 2 41.0 40 995 1 30
130 1 0.0 70 1020 3 25
EOF
cat > 02_single_record.in <<'EOF'
1 3 5
100 2 37.5 45 1001 0 12
EOF
cat > 03_one_station.in <<'EOF'
4 1 1
0 0 10.0 50 1000 1 5
61 0 20.0 60 1005 2 10
122 0 30.0 70 1010 3 15
183 0 40.0 80 1015 4 20
EOF
cat > 04_K_gt_stations.in <<'EOF'
3 5 4
10 0 15.0 50 1000 1 8
20 3 25.0 55 1002 2 9
30 0 35.0 60 1004 3 10
EOF
cat > 05_all_extreme.in <<'EOF'
4 2 2
0 0 45.0 50 1000 0 10
60 1 -3.0 55 1005 1 12
120 0 40.0 60 1010 2 14
180 1 0.0 65 1015 3 16
EOF
cat > 06_interval_tie.in <<'EOF'
4 2 2
0 0 10.0 50 1000 1 5
30 1 12.0 52 1001 1 6
600 0 14.0 54 1002 1 7
630 1 16.0 56 1003 1 8
EOF
cat > 07_hotcold_tie.in <<'EOF'
4 2 3
50 2 30.0 50 1000 1 5
50 1 30.0 51 1001 1 6
20 0 30.0 52 1002 1 7
80 2 5.0 53 1003 1 8
EOF
../gen_weather 1000   5  20   1  08_small_1k.in
../gen_weather 10000  10 100  2  09_medium_10k.in
../gen_weather 100000 10 1000 3  10_large_100k.in
../gen_weather 500    3  500  4  11_many_stations.in
../gen_weather 2000   4  1    5  12_single_station_gen.in
../gen_weather 333    7  13   6  13_odd_sizes.in

for f in *.in; do ../weather_seq "$f" "${f%.in}.out" 2>/dev/null; done
cd ..
echo "Created tests/ with $(ls tests/*.in | wc -l) cases."
