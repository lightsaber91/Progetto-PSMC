#include "driverBFS.c"
#include "cudaBFS.h"

__global__ void kernel_search(gpudata data, csrdata csrg) {

    const int WARPS_PER_BLOCK = blockDim.x / WARP;
    int i, j, warpId;
    UL V, s, e, *neigh;

    *(data.redo) = 0;
    warpId = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP;
    i = warpId;

    while (i < csrg.nv) {
        if (data.queue[i]) {

            data.queue[i] = 0;
            s = csrg.offsets[i];
            e = csrg.offsets[i+1] - s;

            neigh = &csrg.rows[s];

            for (j = threadIdx.x % WARP; j < e; j += WARP) {
                V = neigh[j];
                data.next_queue[V] = 1;
            }
        }
        i += (gridDim.x * blockDim.x) / WARP;
    }
}

__global__ void kernel_compute(gpudata data) {
    UL prev_level;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < data.vertex) {
        if (data.next_queue[tid]) {
            *(data.redo) = 1;
            data.next_queue[tid] = 0;
            prev_level = data.dist[tid];
            data.dist[tid] = data.level+1 < prev_level ? data.level+1 : prev_level;
            data.queue[tid] = (prev_level == ULONG_MAX);
        }
        tid += gridDim.x * blockDim.x;
    }
}


UL *do_bfs_cuda(UL source, csrdata *csrgraph, csrdata *csrgraph_gpu, double *cudatime)
{
    UL U, V, s, e, i, j;

    // Creo le strutture per i timer
    cudaEvent_t exec_start, exec_stop, alloc_copy_start, alloc_copy_stop;
    double alloc_copy_time = 0.0, bfs_time = 0.0;
    char redo = 1;

    // Dati per la gpu
    gpudata host;
    gpudata dev;

    // Variabili per ottimizzare la bfs
    int gpu, num_threads, num_blocks;
    cudaDeviceProp gpu_prop;

    // Leggo le proprietà del device per ottimizzare la bfs
    cudaGetDevice(&gpu);
    cudaGetDeviceProperties(&gpu_prop, gpu);
    num_threads = gpu_prop.maxThreadsPerBlock;
    num_blocks = ceil(csrgraph->nv/num_threads);
    num_blocks = num_blocks < gpu_prop.maxGridSize[0] ? num_blocks : gpu_prop.maxGridSize[0];

    printf("\nNumber of threads: %d\nNumber of blocks: %d\n\n", num_threads, num_blocks);

    // Inizializzo i dati
    host.level = 0;
    host.queue = (char *) Malloc(csrgraph->nv);
    host.dist = (UL *) Malloc(csrgraph->nv*sizeof(UL));
    host.vertex = csrgraph->nv;
    memset(host.queue, 0, csrgraph->nv);
    for (i = 0; i < csrgraph->nv; i++) host.dist[i] = ULONG_MAX;

    // La prima iterazione la faccio seriale
    host.dist[source] = 0;

    // dequeue U
    U = source;
    // Search all neighbors of U
    s = csrgraph->offsets[U]; e = csrgraph->offsets[U+1];
    for (j = s; j < e; j++) {
        V = csrgraph->rows[j];
        // If V is not visited enqueue it and set its distance
        host.queue[V] = 1;
        host.dist[V]  = host.level + 1;
    }
    host.level += 1;
    dev.level = host.level;

    // Inizio ad allocare memoria e copiare i dati sulla gpu
    START_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    copy_data_on_gpu(&host, &dev);
    alloc_copy_time = STOP_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    printf("Time spent for allocation and copy: %.5f\n", alloc_copy_time);

    // Faccio partire il kernel
    START_CUDA_TIMER(&exec_start, &exec_stop);
    while(redo) {
        // lancio il kernel
        kernel_search<<<num_blocks, num_threads>>>(dev, *csrgraph_gpu);
        kernel_compute<<<num_blocks, num_threads>>>(dev);
        dev.level += 1;
        HANDLE_ERROR(cudaMemcpy(&redo, (&dev)->redo, sizeof(char), cudaMemcpyDeviceToHost));
    }
    bfs_time = STOP_CUDA_TIMER(&exec_start, &exec_stop);
    printf("Time spent for cuda bfs: %.5f\n", bfs_time);

    copy_data_on_host(&host, &dev);
    free_gpu_mem(&dev, csrgraph_gpu);

    *cudatime = alloc_copy_time + bfs_time;
    return host.dist;
}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed)
{
    csrdata csrgraph, csrgraph_gpu;     // csr data structure to represent the graph
	FILE *fout;
	UL i;
	UL *dist;             // array of distances from the source

	// Vars for timing
	struct timeval begin, end;
	double cudatime = 0.0, csrtime;
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
    copy_csr_on_gpu(&csrgraph, &csrgraph_gpu);
	END_TIMER(end);
	ELAPSED_TIME(csrtime, begin, end)
	if (randsource) {
		root = random_source(&csrgraph, seed);
		fprintf(stdout, "Random source vertex %lu\n", root);
	}

    dist = do_bfs_cuda(root, &csrgraph, &csrgraph_gpu, &cudatime);

	// Print distance array to file
	fout = Fopen(DISTANCE_OUT_FILE, "w+");
	for (i = 0; i < csrgraph.nv; i++) fprintf(fout, "%lu %lu\n", i, dist[i]);
	fclose(fout);

	// Timing output
    fprintf(stdout, "\n");
	fprintf(stdout, "build csr time = \t%.5f\n", csrtime);
	fprintf(stdout, "Cuda bfs  time = \t%.5f\n", cudatime);
	fprintf(stdout, "\n");

	if(csrgraph.offsets) free(csrgraph.offsets);
	if(csrgraph.rows)    free(csrgraph.rows);

	return dist;
}
