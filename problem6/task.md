# Q6. Connected Components of a Large Graph

- Identify the connected components of a large undirected graph.
- The graph is provided as an adjacency list. Distribute the vertices and their adjacency lists across the processes. Use parallel message-passing techniques to assign each vertex a component ID.

## Constraints

- Number of vertices ( $V$ ):  $1 \leq V \leq 10^5$
- Number of edges ( $E$ ):  $0 \leq E \leq 10^6$
- Vertices are 0-indexed.
- The component ID of a vertex must be the minimum vertex ID present within that connected component.

## Input Format

- First line: A single integer  $V \rightarrow$  number of vertices.
- Next  $V$  lines: The adjacency list of the graph. The  $i$ -th line (where  $0 \leq i < V$ ) has the format:  
 $k \ v_1 \ v_2 \dots v_k$   
where:
  - $k$  = number of neighbors for vertex  $i$
  - $v_j$  = neighbor vertex ID

## Output Format

- Print  $V$  lines. Each line should contain two integers: `vertex_id component_id`.
- The output must be sorted by `vertex_id` in ascending order.

## Sample Input

```
5
1 1
1 0
2 3 4
1 2
1 2
```

### Sample Output

```
0 0
1 0
2 2
3 2
4 2
```

*(Explanation: Vertices 0 and 1 are connected, forming a component with ID 0. Vertices 2, 3, and 4 are connected, forming a component with ID 2).*

## Deliverables for this section

- MPI source code for the assigned algorithm.
- Program must run correctly for  $P = 1, 2, 4, 8$  processes.
- README with compilation and execution instructions.

- Correctness verification against a sequential computation.
- Execution times for  $P = 1, 2, 4, 8$  over varying input sizes.
- Speedup/efficiency plots and a short analysis of communication versus computation.