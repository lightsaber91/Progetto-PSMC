#include "driverBFS.c"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}

static void START_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_ERROR( cudaEventCreate(start));
	HANDLE_ERROR( cudaEventCreate(stop));
	HANDLE_ERROR( cudaEventRecord(*start, 0));
}

static double STOP_AND_PRINT_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_ERROR( cudaEventRecord(*stop, 0));
	HANDLE_ERROR( cudaEventSynchronize(*stop));
	float time=0;
	HANDLE_ERROR( cudaEventElapsedTime(&time, *start, *stop));
	printf("Elapsed Time: %f milliseconds\n", time);
	HANDLE_ERROR( cudaEventDestroy(*start));
	HANDLE_ERROR( cudaEventDestroy(*stop));
	return time;
}

void allocate_memory_on_gpu() {

}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed)
{
	UL *dist;             // array of distances from the source
	csrdata csrgraph;     // csr data structure to represent the graph
	FILE *fout;
	UL i;

	// Vars for timing
	struct timeval begin, end;
	cudaEvent_t start, stop;
	double cudatime, csrtime;
	int timer = 1;

	csrgraph.offsets = NULL;
	csrgraph.rows    = NULL;
	csrgraph.deg     = NULL;

	// Build the CSR data structure
	START_TIMER(begin)
	csrgraph.offsets = (UL *)Malloc((nvertices+1)*sizeof(UL));
	csrgraph.rows    = (UL *)Malloc(nedges       *sizeof(UL));
	csrgraph.deg     = (UL *)Malloc(nvertices    *sizeof(UL));

	build_csr(edges, nedges, nvertices, &csrgraph);
	END_TIMER(end);
	ELAPSED_TIME(csrtime, begin, end)
	if (csrgraph.nv < 50) print_csr(&csrgraph);

	if (randsource) {
		root = random_source(&csrgraph, seed);
		fprintf(stdout, "Random source vertex %lu\n", root);
	}

	allocate_memory_on_gpu();

	// Perform a BFS traversal that returns the array of distances from the source
	START_CUDA_TIMER(&start, &stop);

	// TODO : Implementare la funzione Cuda che esegue BFS

	cudatime = STOP_AND_PRINT_CUDA_TIMER(&start, &stop);

	// Print distance array to file
	fout = Fopen(DISTANCE_OUT_FILE, "w+");
	for (i = 0; i < csrgraph.nv; i++) fprintf(fout, "%lu %lu\n", i, dist[i]);
	fclose(fout);

	// Timing output
	fprintf(stdout, "\n");
	fprintf(stdout, "build csr  time = \t%.5f\n", csrtime);
	fprintf(stdout, "do Cuda OMP time = \t%.5f\n", cudatime);
	fprintf(stdout, "\n");

	if(csrgraph.offsets) free(csrgraph.offsets);
	if(csrgraph.rows)    free(csrgraph.rows);

	return dist;
}
