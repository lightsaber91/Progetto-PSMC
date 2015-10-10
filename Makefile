
CC = gcc
ICC = icc
NVCC = nvcc
CFLAGS = --std=gnu99 -W -Wall -Wextra -O3 --fast-math
ARCH = -march=haswell
#DEBUG = -g
#LDFLAGS = -lm
OMPFLAG = -fopenmp -include ompBFS.h
CUDAFLAG = -include cudaBFS.h
CFILE = driverBFS.c
#OBJFILES = bfs_omp bfs_omp_intel bfs_omp_haswell bfs_omp_intel_haswell bfs_cuda
OBJFILES = clean bfs_omp boh

all: $(OBJFILES)

clean:
	/bin/rm -f $(OBJFILES)

bfs_omp:
	$(CC) $(CFLAGS) $(CFILE) $(OMPFLAG) -o $@

bfs_omp_intel:

boh:
	$(CC) $(ARCH) $(CFLAGS) $(CFILE) $(OMPFLAG) -o $@

bfs_omp_intel_haswell:

bfs_cuda:
