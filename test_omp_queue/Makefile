# Simone Martucci Makefile per bfs con OpenMp e Cuda

CC = gcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CQUEUE = omp_queue.c
DRIVER = driverBFS.c
TARGET = bfs_omp_queue

all: $(TARGET)

bfs_omp_queue: $(CFILE) $(DRIVER)
		$(CC) $(CFLAGS) $(CQUEUE) $(CCOMPFLAG) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
