# Simone Martucci Makefile per bfs con OpenMp e Cuda

CC = gcc
NVCC = nvcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CUDAFLAG = -m64 -arch=sm_35 -O3 -Xcompiler -std=c++98
CFILE = ompBFS.c
CUFILE = cudaBFS.cu
HFILE = cudaBFS.h
CUQUEUE = cuda_queue.cu
HQUEUE = cuda_queue.h lock.h
DRIVER = driverBFS.c
TARGET = bfs_omp bfs_cuda bfs_cuda_queue

all: $(TARGET)

bfs_omp: $(CFILE) $(DRIVER)
	$(CC) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

bfs_cuda: $(CUFILE) $(HFILE) $(DRIVER)
	$(NVCC) $(CUDAFLAG) $(CUFILE) -o $@

bfs_cuda_queue: $(CUQUEUE) $(HQUEUE) $(DRIVER)
		$(NVCC) $(CUDAFLAG) $(CUQUEUE) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
