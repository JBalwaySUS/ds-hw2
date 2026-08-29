# Q6: Connected Components of a Large Graph

## Results Table

Speed-up $S(P) = T_1 / T_P$, calculated from the end-to-end runtime reported
by the MPI program.

| Input size | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|:-----:|:-----:|:-----:|:-----:|
| Small      | 1.00 | 0.88 | 0.06 | 0.06 |
| Medium     | 1.00 | 0.83 | 0.07 | 0.06 |
| Large      | 1.00 | 0.56 | 0.12 | 0.06 |
| Very large | 1.00 | 0.75 | 0.46 | 0.42 |

![Measured speed-up for Q6](speedup.svg)

## Runtime Table

End-to-end runtime in seconds. This includes graph input/distribution,
connected-component computation, and output.

| Input size | Vertices | Undirected edges | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|---------:|-----------------:|------:|------:|------:|------:|
| Small      | 5       | 3         | 0.0007 | 0.0008 | 0.0126 | 0.0125 |
| Medium     | 500     | 2,500     | 0.0010 | 0.0012 | 0.0150 | 0.0165 |
| Large      | 5,000   | 25,000    | 0.0028 | 0.0050 | 0.0234 | 0.0469 |
| Very large | 100,000 | 1,000,000 | 0.0560 | 0.0750 | 0.1211 | 0.1331 |

The program's diagnostic output reports twice the undirected edge count
because it counts adjacency-list entries; every generated edge occurs in both
endpoints' lists.

## Efficiency

Efficiency $E(P) = S(P)/P$, shown as a percentage.

| Input size | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|:-----:|:-----:|:-----:|:-----:|
| Small      | 100.00% | 43.75% | 1.39% | 0.70% |
| Medium     | 100.00% | 41.67% | 1.67% | 0.76% |
| Large      | 100.00% | 28.00% | 2.99% | 0.75% |
| Very large | 100.00% | 37.33% | 11.56% | 5.26% |

## Communication vs. Computation

IPM measured the following percentage of aggregate process wall time inside
MPI calls. The remainder includes local computation, input, and output.

| Input size | $P=1$ | $P=2$ | $P=4$ | $P=8$ |
|------------|:-----:|:-----:|:-----:|:-----:|
| Small      | 0.02% | 0.32% | 8.32% | 7.40% |
| Medium     | 0.02% | 0.31% | 10.71% | 12.46% |
| Large      | 0.03% | 4.23% | 15.30% | 26.93% |
| Very large | 0.77% | 18.27% | 45.80% | 47.35% |

For the very-large case at $P=8$, IPM attributes 17.12% of wall time to
`MPI_Scatterv`, 12.60% to `MPI_Barrier`, 9.29% to `MPI_Bcast`, and 8.33% to
`MPI_Allreduce`. Communication and synchronization therefore consume almost
half of aggregate wall time at the largest process count.

