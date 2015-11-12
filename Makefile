# Simone Martucci Makefile per bfs con OpenMp e Cuda

CC = gcc
NVCC = nvcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CCARCH = -march=haswell --fast-math
CUDAFLAG = -m64 -arch=sm_50 -O3
CFILE = ompBFS.c
CUFILE = cudaBFS.cu
TARGET = omp cuda omp_arch

all: $(TARGET)

clean:
	rm -f $(TARGET) dist_file.dat

omp:
	$(CC) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

omp_arch:
	$(CC) $(CCARCH) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

cuda:
	$(NVCC) $(CUDAFLAG) $(CUFILE) -o $@
