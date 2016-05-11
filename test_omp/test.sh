#!/bin/bash
echo 'launching -g1 -S 20 -E 100'
./bfs_omp -V 1 -g 1 -S 20 -E 100 > graph1_S20_E100.txt
echo 'done -g1 -S 20 -E 100'
echo 'launching -g1 -S 20 -E 200'
./bfs_omp -V 1 -g 1 -S 20 -E 200 > graph1_S20_E200.txt
echo 'done -g1 -S 20 -E 200'
echo 'launching -g1 -S 20 -E 400'
./bfs_omp -V 1 -g 1 -S 20 -E 400 > graph1_S20_E400.txt
echo 'done -g1 -S 20 -E 400'
#echo 'launching -g1 -S 20 -E 800'
#./bfs_omp -V 1 -g 1 -S 20 -E 800 > graph1_S20_E800.txt
#echo 'done -g1 -S 20 -E 800'

echo 'launching -g2 -S 20 -E 100'
./bfs_omp -V 1 -g 2 -S 20 -E 100 > graph2_S20_E100.txt
echo 'done -g2 -S 20 -E 100'
echo 'launching -g2 -S 20 -E 100'
./bfs_omp -V 1 -g 2 -S 20 -E 200 > graph2_S20_E200.txt
echo 'done -g2 -S 20 -E 200'
echo 'launching -g2 -S 20 -E 400'
./bfs_omp -V 1 -g 2 -S 20 -E 400 > graph2_S20_E400.txt
echo 'done -g2 -S 20 -E 400'
echo 'launching -g2 -S 20 -E 800'
./bfs_omp -V 1 -g 2 -S 20 -E 800 > graph2_S20_E800.txt
echo 'done -g2 -S 20 -E 800'
