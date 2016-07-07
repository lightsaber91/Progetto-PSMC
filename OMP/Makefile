# Simone Martucci Makefile per bfs con OpenMp

CC = gcc
CFLAGS = -std=gnu99 -W -Wall -Wextra -O3 -fopenmp
CFILE = ompBFS.c
CQUEUE = omp_queue.c
SOURCE = ../source/driverBFS.c
TARGET = omp queue_omp

all: $(TARGET)

omp: $(CFILE) $(SOURCE)
	$(CC) $(CFLAGS) $(CFILE) $(CCOMPFLAG) -o ../$@

queue_omp: $(CFILE) $(SOURCE)
		$(CC) $(CFLAGS) $(CQUEUE) $(CCOMPFLAG) -o ../$@

.PHONY: clean

clean:
	for file in $(TARGET); do \
		rm -f ../$$file dist_file.dat; \
	done
