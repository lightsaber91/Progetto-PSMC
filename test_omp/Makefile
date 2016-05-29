# Simone Martucci Makefile per bfs con OpenMp e Cuda

CC = gcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CFILE = ompBFS.c
DRIVER = driverBFS.c
TARGET = bfs_omp

all: $(TARGET)

bfs_omp: $(CFILE) $(DRIVER)
	$(CC) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) dist_file.dat
