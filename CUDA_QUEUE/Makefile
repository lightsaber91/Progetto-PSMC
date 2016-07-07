# Simone Martucci Makefile per bfs con Cuda

NVCC = nvcc
CUDAFLAG = -m64 -arch=sm_35 -O3 -Xcompiler -std=c++98
CUFILE = cuda_queue_kernel.cu
SOURCE = ../source/driverBFS.c ../source/cuda_error.h ../source/cuda_csr.h ../source/cuda_timer.h cuda_queue_data.h cuda_queue_utils.h
TARGET = cuda_queue_bfs

all: $(TARGET)

cuda_queue_bfs: $(CUFILE) $(SOURCE)
	$(NVCC) $(CUDAFLAG) $(CUFILE) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
