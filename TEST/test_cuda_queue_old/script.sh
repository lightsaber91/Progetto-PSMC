#!/bin/sh

echo 'BFS CUDA G1 - S 20 E 50'
./bfs_cuda_no_exch -V 1 -g 1 -S 20 -E 50 > g1_S20_E50
echo 'BFS CUDA G1 - S 20 E 150'
./bfs_cuda_no_exch -V 1 -g 1 -S 20 -E 150 > g1_S20_E150
echo 'BFS CUDA G1 - S 20 E 250'
./bfs_cuda_no_exch -V 1 -g 1 -S 20 -E 250 > g1_S20_E250
echo 'BFS CUDA G1 - S 20 E 350'
./bfs_cuda_no_exch -V 1 -g 1 -S 20 -E 350 > g1_S20_E350


echo 'BFS CUDA G2 - S 20 E 50'
./bfs_cuda_no_exch -V 1 -g 2 -S 20 -E 50 > g2_S20_E50
echo 'BFS CUDA G2 - S 20 E 150'
./bfs_cuda_no_exch -V 1 -g 2 -S 20 -E 150 > g2_S20_E150
echo 'BFS CUDA G2 - S 20 E 250'
./bfs_cuda_no_exch -V 1 -g 2 -S 20 -E 250 > g2_S20_E250
echo 'BFS CUDA G2 - S 20 E 350'
./bfs_cuda_no_exch -V 1 -g 2 -S 20 -E 350 > g2_S20_E350

echo 'BFS CUDA G1 - S 21 E 20'
./bfs_cuda_no_exch -V 1 -g 1 -S 21 -E 20 > g1_S21_E20
echo 'BFS CUDA G1 - S 22 E 20'
./bfs_cuda_no_exch -V 1 -g 1 -S 22 -E 20 > g1_S22_E20
echo 'BFS CUDA G1 - S 23 E 20'
./bfs_cuda_no_exch -V 1 -g 1 -S 23 -E 20 > g1_S23_E20
echo 'BFS CUDA G1 - S 24 E 20'
./bfs_cuda_no_exch -V 1 -g 1 -S 24 -E 20 > g1_S24_E20

echo 'BFS CUDA G2 - S 21 E 20'
./bfs_cuda_no_exch -V 1 -g 2 -S 21 -E 20 > g2_S21_E20
echo 'BFS CUDA G2 - S 22 E 20'
./bfs_cuda_no_exch -V 1 -g 2 -S 22 -E 20 > g2_S22_E20
echo 'BFS CUDA G2 - S 23 E 20'
./bfs_cuda_no_exch -V 1 -g 2 -S 23 -E 20 > g2_S23_E20
echo 'BFS CUDA G2 - S 24 E 20'
./bfs_cuda_no_exch -V 1 -g 2 -S 24 -E 20 > g2_S24_E20
