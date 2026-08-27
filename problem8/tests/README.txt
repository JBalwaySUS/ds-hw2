Q8 Weather Analytics - test suite
=================================

Each test is a pair:
  NAME.in   -> input dataset (format: "N K S", then N measurement lines)
  NAME.out  -> the correct output, pre-computed by the verified sequential program

run_tests.sh runs the MPI program on every .in for P = 1, 2, 4, 8 and checks that
its output exactly matches the .out. A test PASSES only if all four process counts
produce the identical, correct result.

Cases and what each one stresses
--------------------------------
01_tiny_sample        5 records  hand-verified reference values
02_single_record      N=1        smallest possible input
03_one_station        S=1        all records on one station
04_K_gt_stations      K>#stations Top-K must list only stations present
05_all_extreme        every temp >=40 or <=0  extreme-event counting
06_interval_tie       two intervals tie on count -> smaller interval id wins
07_hotcold_tie        temps tie -> tie-break by smaller timestamp, then station id
08_small_1k           N=1000      general small
09_medium_10k         N=10000     general medium
10_large_100k         N=100000    general large
11_many_stations      S near N    many distinct stations
12_single_station_gen S=1, N=2000 single station, larger
13_odd_sizes          N=333,S=13  awkward sizes

How to run (on the cluster)
---------------------------
1. Put this tests/ folder inside your Q8 folder (so ../weather_mpi exists after compiling).
2. Compile:  mpicxx -O2 -std=c++17 -o weather_mpi weather_mpi.cpp
3. Get a compute node:
   salloc --nodes=1 --ntasks=4 --time=00:15:00 --partition=debug srun --pty bash
4. cd tests && bash run_tests.sh
5. Look for "ALL TESTS PASSED".

Regenerating expected outputs (only if you change the input files)
------------------------------------------------------------------
  g++ -O2 -std=c++17 -o ../weather_seq ../weather_seq.cpp
  for f in *.in; do ../weather_seq "$f" "${f%.in}.out"; done
