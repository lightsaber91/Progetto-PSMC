# Simone Martucci Makefile per bfs con OpenMp e Cuda
CC = gcc
ICC = icc
NVCC = nvcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3
CCARCH = -march=haswell --fast-math
IARCH = -march=core-avx2 -xCORE-AVX2 -fma -ipo
CCOMPFLAG = -fopenmp -include ompBFS.h
IOMPFLAG = -openmp -include ompBFS.h
CUDAFLAG = -m64 -arch=sm_50 -O3 -include cudaBFS.h
CFILE = driverBFS.c
TARGET = bfs_gcc bfs_cuda bfs_intel bfs_gcc_arch bfs_intel_arch

all: $(TARGET)

clean:
	rm -f $(TARGET)

bfs_gcc:
	$(CC) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

bfs_intel:
	$(ICC) $(CFLAGS) $(CFILE) $(IOMPFLAG) -o $@

bfs_gcc_arch:
	$(CC) $(CCARCH) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

bfs_intel_arch:
	$(ICC) $(IARCH) $(CFLAGS) $(CFILE) $(IOMPFLAG) -o $@

bfs_cuda:
	$(NVCC) $(CUDAFLAG) $(CFILE) -o $@
