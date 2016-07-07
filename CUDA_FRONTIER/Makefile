# Simone Martucci Makefile per bfs con OpenMp e Cuda

NVCC = nvcc
CUDAFLAG = -m64 -arch=sm_35 -O3 -Xcompiler -std=c++98
CUFILE = cuda_frontier_kernel.cu
SOURCE = ../source/driverBFS.c ../source/cuda_error.h ../source/cuda_csr.h ../source/cuda_timer.h cuda_frontier_data.h
TARGET = cuda_frontier_bfs

all: $(TARGET)

cuda_frontier_bfs: $(CUFILE) $(SOURCE)
	$(NVCC) $(CUDAFLAG) $(CUFILE) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
