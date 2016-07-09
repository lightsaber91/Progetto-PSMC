# Simone Martucci Makefile per bfs con OpenMp

CC = gcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CFILE = ompBFS.c
CQUEUE = omp_queue.c
TARGET = omp queue_omp

all: $(TARGET)

omp: $(CFILE)
	$(CC) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o ../$@

queue_omp: $(CQUEUE)
	$(CC) $(CFLAGS) $(CQUEUE) $(CCOMPFLAG) -o ../$@

.PHONY: clean

clean:
	rm -f ../omp ../queue_omp dist_file.dat
