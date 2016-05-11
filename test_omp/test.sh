#!/bin/bash

echo 'launching -g1 -S 25 -E 20'
./bfs_omp -V 1 -g 1 -S 25 -E 20 > graph1_E20_S25.txt
echo 'done -g1 -S 25 -E 20'

echo 'launching -g1 -S 24 -E 20'
./bfs_omp -V 1 -g 1 -S 24 -E 20 > graph1_E20_S24.txt
echo 'done -g1 -S 24 -E 20'

echo 'launching -g1 -S 23 -E 20'
./bfs_omp -V 1 -g 1 -S 23 -E 20 > graph1_E20_S23.txt
echo 'done -g1 -S 23 -E 20'

echo 'launching -g1 -S 22 -E 20'
./bfs_omp -V 1 -g 1 -S 22 -E 20 > graph1_E20_S22.txt
echo 'done -g1 -S 22 -E 20'

echo 'launching -g2 -S 25 -E 20'
./bfs_omp -V 1 -g 2 -S 25 -E 20 > graph2_E20_S25.txt
echo 'done -g2 -S 25 -E 20'

echo 'launching -g2 -S 24 -E 20'
./bfs_omp -V 1 -g 2 -S 24 -E 20 > graph2_E20_S24.txt
echo 'done -g2 -S 24 -E 20'

echo 'launching -g1 -S 23 -E 20'
./bfs_omp -V 1 -g 2 -S 23 -E 20 > graph2_E20_S23.txt
echo 'done -g2 -S 23 -E 20'

echo 'launching -g2 -S 22 -E 20'
./bfs_omp -V 1 -g 2 -S 22 -E 20 > graph2_E20_S22.txt
echo 'done -g2 -S 22 -E 20'

