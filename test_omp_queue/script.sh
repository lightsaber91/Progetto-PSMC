#!/bin/sh

echo 'G1 S20 E100'
./bfs_omp_queue -V 1 -g 1 -S20 -E 100 > ./S_20/graph_1_S20_E100
echo 'G1 S20 E200'
./bfs_omp_queue -V 1 -g 1 -S20 -E 200 > ./S_20/graph_1_S20_E200
echo 'G1 S20 E400'
./bfs_omp_queue -V 1 -g 1 -S20 -E 400 > ./S_20/graph_1_S20_E400
echo 'G1 S20 E800'
./bfs_omp_queue -V 1 -g 1 -S20 -E 800 > ./S_20/graph_1_S20_E800

echo 'G2 S20 E100'
./bfs_omp_queue -V 1 -g 2 -S20 -E 100 > ./S_20/graph_2_S20_E100
echo 'G2 S20 E200'
./bfs_omp_queue -V 1 -g 2 -S20 -E 200 > ./S_20/graph_2_S20_E200
echo 'G2 S20 E400'
./bfs_omp_queue -V 1 -g 2 -S20 -E 400 > ./S_20/graph_2_S20_E400
echo 'G2 S20 E800'
./bfs_omp_queue -V 1 -g 2 -S20 -E 800 > ./S_20/graph_2_S20_E800

echo 'G1 E20 S22'
./bfs_omp_queue -V 1 -g 1 -S22 -E 20 > ./E_20/graph_1_E20_S22
echo 'G1 E20 S23'
./bfs_omp_queue -V 1 -g 1 -S23 -E 20 > ./E_20/graph_1_E20_S23
echo 'G1 E20 S24'
./bfs_omp_queue -V 1 -g 1 -S24 -E 20 > ./E_20/graph_1_E20_S24
echo 'G1 E20 S25'
./bfs_omp_queue -V 1 -g 1 -S25 -E 20 > ./E_20/graph_1_E20_S25

echo 'G2 E20 S22'
./bfs_omp_queue -V 1 -g 2 -S22 -E 20 > ./E_20/graph_2_E20_S22
echo 'G2 E20 S23'
./bfs_omp_queue -V 1 -g 2 -S23 -E 20 > ./E_20/graph_2_E20_S23
echo 'G2 E20 S24'
./bfs_omp_queue -V 1 -g 2 -S24 -E 20 > ./E_20/graph_2_E20_S24
echo 'G2 E20 S25'
./bfs_omp_queue -V 1 -g 2 -S25 -E 20 > ./E_20/graph_2_E20_S25
