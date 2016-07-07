# Simone Martucci Makefile per bfs con OpenMp e Cuda

SUBDIRS = OMP CUDA_FRONTIER CUDA_QUEUE

all:
	for dir in $(SUBDIRS); do \
        $(MAKE) -C $$dir; \
    done

.PHONY: clean

clean:
	for dir in $(SUBDIRS); do \
        $(MAKE) -C $$dir clean; \
    done
