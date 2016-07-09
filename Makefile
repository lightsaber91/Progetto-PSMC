# Simone Martucci Makefile per bfs con OpenMp e Cuda

CLEAN = omp_clean frontier_clean queue_clean
TARGET = omp queue frontier

all: $(TARGET)

omp:
	$(MAKE) -C OMP;

queue:
	$(MAKE) -C CUDA_QUEUE;

frontier:
	$(MAKE) -C CUDA_FRONTIER;

.PHONY: clean

clean: $(CLEAN)

omp_clean:
	$(MAKE) -C OMP clean;

frontier_clean:
	$(MAKE) -C CUDA_FRONTIER clean;

queue_clean:
	$(MAKE) -C CUDA_QUEUE clean;
