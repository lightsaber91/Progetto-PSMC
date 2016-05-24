#!/bin/sh

echo 'G1 20 - 50 AUTO'
./bfs_cuda -V 1 -g 1 -S 20 -E 50 >> g1_S20_E50_auto
echo 'G1 20 - 100 AUTO'
./bfs_cuda -V 1 -g 1 -S 20 -E 100 >> g1_S20_E100_auto
echo 'G1 20 - 150 AUTO'
./bfs_cuda -V 1 -g 1 -S 20 -E 150 >> g1_S20_E150_auto
echo 'G1 20 - 200 AUTO'
./bfs_cuda -V 1 -g 1 -S 20 -E 200 >> g1_S20_E200_auto

echo 'G2 20 - 50 AUTO'
./bfs_cuda -V 1 -g 2 -S 20 -E 50 >> g2_S20_E50_auto
echo 'G2 20 - 100 AUTO'
./bfs_cuda -V 1 -g 2 -S 20 -E 100 >> g2_S20_E100_auto
echo 'G2 20 - 150 AUTO'
./bfs_cuda -V 1 -g 2 -S 20 -E 150 >> g2_S20_E150_auto
echo 'G2 20 - 200 AUTO'
./bfs_cuda -V 1 -g 2 -S 20 -E 200 >> g2_S20_E200_auto

echo 'G1 20 - 50 - 128'
./bfs_cuda -V 1 -g 1 -S 20 -E 50 -T 128 >> g1_S20_E50_128
echo 'G1 20 - 100 - 128'
./bfs_cuda -V 1 -g 1 -S 20 -E 100 -T 128 >> g1_S20_E100_128
echo 'G1 20 - 150 - 128'
./bfs_cuda -V 1 -g 1 -S 20 -E 150 -T 128 >> g1_S20_E150_128
echo 'G1 20 - 200 - 128'
./bfs_cuda -V 1 -g 1 -S 20 -E 200 -T 128 >> g1_S20_E200_128

echo 'G2 20 - 50 - 128'
./bfs_cuda -V 1 -g 2 -S 20 -E 50 -T 128 >> g2_S20_E50_128
echo 'G2 20 - 100 - 128'
./bfs_cuda -V 1 -g 2 -S 20 -E 100 -T 128 >> g2_S20_E100_128
echo 'G2 20 - 150 - 128'
./bfs_cuda -V 1 -g 2 -S 20 -E 150 -T 128 >> g2_S20_E150_128
echo 'G2 20 - 200 - 128'
./bfs_cuda -V 1 -g 2 -S 20 -E 200 -T 128 >> g2_S20_E200_128

echo 'G1 20 - 50 - 256'
./bfs_cuda -V 1 -g 1 -S 20 -E 50 -T 256 >> g1_S20_E50_256
echo 'G1 20 - 100 - 256'
./bfs_cuda -V 1 -g 1 -S 20 -E 100 -T 256 >> g1_S20_E100_256
echo 'G1 20 - 150 - 256'
./bfs_cuda -V 1 -g 1 -S 20 -E 150 -T 256 >> g1_S20_E150_256
echo 'G1 20 - 200 - 256'
./bfs_cuda -V 1 -g 1 -S 20 -E 200 -T 256 >> g1_S20_E200_256

echo 'G2 20 - 50 - 256'
./bfs_cuda -V 1 -g 2 -S 20 -E 50 -T 256 >> g2_S20_E50_256
echo 'G2 20 - 100 - 256'
./bfs_cuda -V 1 -g 2 -S 20 -E 100 -T 256 >> g2_S20_E100_256
echo 'G2 20 - 150 - 256'
./bfs_cuda -V 1 -g 2 -S 20 -E 150 -T 256 >> g2_S20_E150_256
echo 'G2 20 - 200 - 256'
./bfs_cuda -V 1 -g 2 -S 20 -E 200 -T 256 >> g2_S20_E200_256

echo 'G1 20 - 50 - 512'
./bfs_cuda -V 1 -g 1 -S 20 -E 50 -T 512 >> g1_S20_E50_512
echo 'G1 20 - 100 - 512'
./bfs_cuda -V 1 -g 1 -S 20 -E 100 -T 512 >> g1_S20_E100_512
echo 'G1 20 - 150 - 512'
./bfs_cuda -V 1 -g 1 -S 20 -E 150 -T 512 >> g1_S20_E150_512
echo 'G1 20 - 200 - 512'
./bfs_cuda -V 1 -g 1 -S 20 -E 200 -T 512 >> g1_S20_E200_512

echo 'G2 20 - 50 - 512'
./bfs_cuda -V 1 -g 2 -S 20 -E 50 -T 512 >> g2_S20_E50_512
echo 'G2 20 - 100 - 512'
./bfs_cuda -V 1 -g 2 -S 20 -E 100 -T 512 >> g2_S20_E100_512
echo 'G2 20 - 150 - 512'
./bfs_cuda -V 1 -g 2 -S 20 -E 150 -T 512 >> g2_S20_E150_512
echo 'G2 20 - 200 - 512'
./bfs_cuda -V 1 -g 2 -S 20 -E 200 -T 512 >> g2_S20_E200_512

echo 'G1 20 - 50 - 1024'
./bfs_cuda -V 1 -g 1 -S 20 -E 50 -T 1024 >> g1_S20_E50_1024
echo 'G1 20 - 100 - 1024'
./bfs_cuda -V 1 -g 1 -S 20 -E 100 -T 1024 >> g1_S20_E100_1024
echo 'G1 20 - 150 - 1024'
./bfs_cuda -V 1 -g 1 -S 20 -E 150 -T 1024 >> g1_S20_E150_1024
echo 'G1 20 - 200 - 1024'
./bfs_cuda -V 1 -g 1 -S 20 -E 200 -T 1024 >> g1_S20_E200_1024

echo 'G2 20 - 50 - 1024'
./bfs_cuda -V 1 -g 2 -S 20 -E 50 -T 1024 >> g2_S20_E50_1024
echo 'G2 20 - 100 - 1024'
./bfs_cuda -V 1 -g 2 -S 20 -E 100 -T 1024 >> g2_S20_E100_1024
echo 'G2 20 - 150 - 1024'
./bfs_cuda -V 1 -g 2 -S 20 -E 150 -T 1024 >> g2_S20_E150_1024
echo 'G2 20 - 200 - 1024'
./bfs_cuda -V 1 -g 2 -S 20 -E 200 -T 1024 >> g2_S20_E200_1024
