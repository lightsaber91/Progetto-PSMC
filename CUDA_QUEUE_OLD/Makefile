# Simone Martucci Makefile per bfs con Cuda

NVCC = nvcc
CUDAFLAG = -m64 -arch=sm_35 -O3 -Xcompiler -std=c++98
CUFILE = cuda_queue_kernel.cu
SOURCE = cuda_queue_data.h cuda_queue_utils.h
TARGET = cuda_queue_old

all: $(TARGET)

cuda_queue_old: $(CUFILE) $(SOURCE)
	$(NVCC) $(CUDAFLAG) $(CUFILE) $(CCOMPFLAG) -o ../$@

.PHONY: clean

clean:
	rm -f ../$(TARGET) dist_file.dat
