# Simone Martucci Makefile per bfs con OpenMp e Cuda

CC = gcc
NVCC = nvcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CUDAFLAG = -m64 -arch=sm_50 -O3 -Xcompiler -std=c++98
CUFILE = cudaBFS.cu
HFILE = cudaBFS.h
DRIVER = driverBFS.c
TARGET = bfs_cuda

all: $(TARGET)

bfs_cuda: $(CUFILE) $(HFILE) $(DRIVER)
	$(NVCC) $(CUDAFLAG) $(CUFILE) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
