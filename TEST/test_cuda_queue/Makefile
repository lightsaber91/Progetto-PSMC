# Simone Martucci Makefile per bfs con OpenMp e Cuda

NVCC = nvcc
CUDAFLAG = -m64 -arch=sm_35 -O3 -Xcompiler -std=c++98
CFILE = ompBFS.c
CUQUEUE = cuda_queue.cu
HQUEUE = cuda_queue.h
DRIVER = driverBFS.c
TARGET = bfs_cuda_no_exch

all: $(TARGET)

bfs_cuda_no_exch: $(CUQUEUE) $(HQUEUE) $(DRIVER)
		$(NVCC) $(CUDAFLAG) $(CUQUEUE) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
