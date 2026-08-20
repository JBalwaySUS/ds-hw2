# Q1. Distributed Matrix Multiplication using the Row-Row Method

You are given two matrices,  $A$  (of dimension  $m \times n$ ) and  $B$  (of dimension  $n \times p$ ). Implement an MPI program that computes  $C = A \times B$  using the **Row-Row method**. In this method, matrix  $A$  is partitioned row-wise across processes, and matrix  $B$  is fully replicated (broadcast) to every process. Each row of  $C$  is computed as a weighted sum of the rows of  $B$ , where the weights are the entries of the corresponding row of  $A$ :

$$c_i = a_i[1] \cdot B[1, :] + a_i[2] \cdot B[2, :] + \dots + a_i[n] \cdot B[n, :]$$

## Input Constraints

- $m, n, p$  range from small (e.g.  $3 \times 3$ , for manual verification) to large (e.g.  $1000 \times 1000$  or more, for timing/scaling)
- Matrices need not be square; test skewed shapes (tall  $m \gg n$ , or wide  $n \gg m$ )
- Test both cases:  $m$  divisible by  $P$ , and  $m$  not divisible by  $P$
- Edge cases:  $m = 1$  (single row),  $n = 1$  (single column in  $A$  / single row in  $B$ )
- Entries are Integers.

## Matrix Layout

- Master reads/generates  $A$  ( $m \times n$ ) and  $B$  ( $n \times p$ ).
- $A$  is partitioned horizontally: each of  $P$  workers gets  $\sim m/P$  rows (handle  $m$  not divisible by  $P$ ).
- $B$  is broadcast in full to all workers.

## Computation

- Workers compute their assigned row-slices of  $C$  independently, without inter-worker communication.

## Collection

- Master gathers all row-slices from workers and reconstructs  $C$ .

## Example 1 — even split, $P = 3$

$$A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \\ -1 & 4 \end{pmatrix}_{3 \times 2} \quad B = \begin{pmatrix} 2 & 3 & 4 \\ 1 & 0 & -1 \end{pmatrix}_{2 \times 3}$$

Distribution: P0  $\rightarrow$  row 1, P1  $\rightarrow$  row 2, P2  $\rightarrow$  row 3

$$c_1 = 1 \cdot (2, 3, 4) + 2 \cdot (1, 0, -1) = (4, 3, 2)$$

$$c_2 = 0 \cdot (2, 3, 4) + 3 \cdot (1, 0, -1) = (3, 0, -3)$$

$$c_3 = -1 \cdot (2, 3, 4) + 4 \cdot (1, 0, -1) = (2, -3, -8)$$

$$C = \begin{pmatrix} 4 & 3 & 2 \\ 3 & 0 & -3 \\ 2 & -3 & -8 \end{pmatrix}_{3 \times 3}$$

## Example 2 — uneven split, $P = 3$

$$A = \begin{pmatrix} 1 & 0 \\ 2 & -1 \\ 0 & 3 \\ 1 & 1 \end{pmatrix}_{4 \times 2} \quad B = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}_{2 \times 2}$$

Distribution: P0  $\rightarrow$  2 rows, P1  $\rightarrow$  1 row, P2  $\rightarrow$  1 row

$$c_1 = (1, 2) \quad c_2 = (2, 3) \quad c_3 = (0, 3) \quad c_4 = (1, 3)$$

$$C = \begin{pmatrix} 1 & 2 \\ 2 & 3 \\ 0 & 3 \\ 1 & 3 \end{pmatrix}_{4 \times 2}$$