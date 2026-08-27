Q1 Row-Row MatMul — test suite
==============================

Each test is a pair:
  NAME.in   -> input matrices (format: "m n p", then A row-major, then B row-major)
  NAME.out  -> the correct result C, pre-computed by the verified sequential program

run_tests.sh runs the MPI program on every .in for P = 1, 2, 4, 8 and checks that
its output exactly matches the .out. A test PASSES only if all four process counts
produce the identical, correct matrix.

Cases and what each one stresses
--------------------------------
01_pdf_example1     3x2x3   Example 1 from the assignment PDF (known answer)
02_pdf_example2     4x2x2   Example 2 from the assignment PDF (uneven split, known answer)
03_one_by_one       1x1x1   Smallest possible; scalar multiply
04_identity         4x4x4   A x I = A  (easy human check)
05_zeros            4x4x3   Many zero entries; exercises the zero-skip path
06_even_split       8x4x5   m divisible by 2, 4 and 8 (clean Scatterv)
07_uneven_split     7x5x4   m NOT divisible by 2 or 4 (Scatterv remainder rows)
08_single_row       1x6x5   m = 1 (edge case)
09_n_equals_1       5x1x4   n = 1 (single column-of-A / row-of-B)
10_p_equals_1       5x4x1   p = 1 (single output column)
11_tall             30x3x4  m >> n (tall, skewed shape)
12_wide             3x30x4  n >> m (wide, skewed shape)
13_fewer_rows_than_P 3x5x5  m < P: at P=4/8 some ranks receive 0 rows
14_odd_dims         13x11x7 Awkward prime dimensions
15_square           10x10x10 Plain square case
16_medium           100x80x60 Larger case, still exact-checked against sequential

How to run (on the cluster)
---------------------------
1. Put this tests/ folder inside your Q1 folder (so ../rowrow exists after compiling).
2. Compile the MPI program:   mpicxx -O2 -std=c++17 -o rowrow rowrow_matmul.cpp
3. Get a compute node:        salloc --nodes=1 --ntasks=4 --time=00:10:00 --partition=debug srun --pty bash
4. cd tests && bash run_tests.sh
5. Look for "ALL TESTS PASSED".

Regenerating expected outputs (only if you change the input files)
------------------------------------------------------------------
  g++ -O2 -std=c++17 -o ../seq ../seq_matmul.cpp
  for f in *.in; do ../seq "$f" "${f%.in}.out"; done
