# Simone Martucci Makefile per bfs con Cuda

NVCC = nvcc
CUDAFLAG = -m64 -arch=sm_35 -O3 -Xcompiler -std=c++98
CUFILE = cuda_frontier_kernel.cu
SOURCE = cuda_frontier_data.h
TARGET = cuda_frontier_bfs

all: $(TARGET)

cuda_frontier_bfs: $(CUFILE) $(SOURCE)
	$(NVCC) $(CUDAFLAG) $(CUFILE) -o ../$@

.PHONY: clean

clean:
	rm -f ../$(TARGET) dist_file.dat
