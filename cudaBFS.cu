#include "driverBFS.c"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        //exit( EXIT_FAILURE );
    }
}

static void START_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop)
{
	HANDLE_ERROR( cudaEventCreate(start));
	HANDLE_ERROR( cudaEventCreate(stop));
	HANDLE_ERROR( cudaEventRecord(*start, 0));
}

static double STOP_AND_PRINT_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop)
{
	HANDLE_ERROR( cudaEventRecord(*stop, 0));
	HANDLE_ERROR( cudaEventSynchronize(*stop));
	float time=0;
	HANDLE_ERROR( cudaEventElapsedTime(&time, *start, *stop));
	HANDLE_ERROR( cudaEventDestroy(*start));
	HANDLE_ERROR( cudaEventDestroy(*stop));
	return time;
}

__global__ void cuda_bfs_parallel()
{
    return;
}

UL *do_bfs_cuda(UL source, csrdata *csrg, double *cudatime)
{
    UL *d_cpu = NULL, *d_gpu = NULL, i;
    int *v_cpu = NULL, *v_gpu = NULL;
    cudaEvent_t start, stop;

    // Qui verranno allocate tutte le strutture dati da passare alla gpu
    d_cpu = (UL *)Malloc(csrg->nv*sizeof(UL));
    v_cpu = (int *)Malloc(csrg->nv*sizeof(int));

    memset(v_cpu, 0, csrg->nv*sizeof(int));
    for (i = 0; i < csrg->nv; i++) d_cpu[i] = ULONG_MAX;

    HANDLE_ERROR(cudaMalloc((void**)&d_gpu, csrg->nv*sizeof(UL)));
    HANDLE_ERROR(cudaMalloc((void**)&v_gpu, csrg->nv*sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_gpu, d_cpu, csrg->nv*sizeof(UL), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(v_gpu, v_cpu, csrg->nv*sizeof(int), cudaMemcpyHostToDevice));

    START_CUDA_TIMER(&start, &stop);
    //faccio partire il kernel
    cuda_bfs_parallel<<<1,1>>>();
    *cudatime = STOP_AND_PRINT_CUDA_TIMER(&start, &stop);

    // Qui copio il risultato sulla cpu
    HANDLE_ERROR(cudaMemcpy(d_cpu, d_gpu, csrg->nv*sizeof(UL), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(v_cpu, v_gpu, csrg->nv*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_gpu));
    HANDLE_ERROR(cudaFree(v_gpu));

    return d_cpu;
}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed)
{
    csrdata csrgraph;     // csr data structure to represent the graph
	FILE *fout;
	UL i;
	UL *dist;             // array of distances from the source

	// Vars for timing
	struct timeval begin, end;
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

    dist = do_bfs_cuda(root, &csrgraph, &cudatime);

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
