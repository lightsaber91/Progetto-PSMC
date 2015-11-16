# Simone Martucci Makefile per bfs con OpenMp e Cuda

CC = gcc
NVCC = nvcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CCARCH = -march=haswell --fast-math
CUDAFLAG = -m64 -arch=sm_50 -O3
CFILE = ompBFS.c
CUFILE = cudaBFS.cu
DRIVER = driverBFS.c
TARGET = bfs_omp bfs_cuda bfs_omp_arch

all: $(TARGET)

bfs_omp: $(CFILE) $(DRIVER)
	$(CC) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

bfs_omp_arch: $(CFILE) $(DRIVER)
	$(CC) $(CCARCH) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

bfs_cuda: $(CUFILE) $(DRIVER)
	$(NVCC) $(CUDAFLAG) $(CUFILE) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
