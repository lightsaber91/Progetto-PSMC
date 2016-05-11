#!/bin/bash
./bfs_omp -V 1 -g 1 -S 20 -E 800 > graph1_S20_E800.txt
./bfs_omp -V 1 -g 1 -S 27 -E 10 > graph1_S27_E10.txt

./bfs_omp -V 1 -g 2 -S 20 -E 800 > graph2_S20_E800.txt
./bfs_omp -V 1 -g 2 -S 27 -E 10 > graph2_S27_E10.txt
