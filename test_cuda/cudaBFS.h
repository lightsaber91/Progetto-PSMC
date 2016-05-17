typedef struct {
    char *redo;
    char *queue;
    char *frontier;
    int warp_size;
    UL *dist;
    UL vertex;
    UL level;
} gpudata;


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static inline void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}

static inline void START_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop)
{
	HANDLE_ERROR(cudaEventCreate(start));
	HANDLE_ERROR(cudaEventCreate(stop));
	HANDLE_ERROR(cudaEventRecord(*start, 0));
}

static inline double STOP_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop)
{
    float time=0;
	HANDLE_ERROR(cudaEventRecord(*stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(*stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, *start, *stop));
	HANDLE_ERROR(cudaEventDestroy(*start));
	HANDLE_ERROR(cudaEventDestroy(*stop));
	return (time/1000.0);
}


inline void copy_csr_on_gpu(const csrdata *csrg, csrdata *csrg_gpu){
    UL nv = csrg->nv;
    UL ne = csrg->ne;

    HANDLE_ERROR(cudaMalloc((void**) &csrg_gpu->offsets, (nv+1) * sizeof(UL)));
    HANDLE_ERROR(cudaMalloc((void**) &csrg_gpu->rows, ne * sizeof(UL)));

    HANDLE_ERROR(cudaMemcpy(csrg_gpu->offsets, csrg->offsets, (nv+1) * sizeof(UL), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(csrg_gpu->rows, csrg->rows, ne * sizeof(UL), cudaMemcpyHostToDevice));

    csrg_gpu->nv = nv;
    csrg_gpu->ne = ne;
}

inline void copy_data_on_gpu(const gpudata *host, gpudata *gpu) {
    gpu->vertex = host->vertex;

    HANDLE_ERROR(cudaMalloc((void**) &gpu->redo, sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->queue, host->vertex * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->frontier, host->vertex * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->dist, host->vertex * sizeof(UL)));

    HANDLE_ERROR(cudaMemcpy(gpu->queue, host->queue, host->vertex * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->dist, host->dist, host->vertex * sizeof(UL), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemset(gpu->frontier, 0, host->vertex * sizeof(char)));
}

void copy_data_on_host(gpudata *host, gpudata *gpu) {
    HANDLE_ERROR(cudaMemcpy(host->dist, gpu->dist, (host->vertex) * sizeof(UL),cudaMemcpyDeviceToHost));
}

void free_gpu_mem(gpudata *gpu) {
    HANDLE_ERROR(cudaFree(gpu->queue));
    HANDLE_ERROR(cudaFree(gpu->frontier));
    HANDLE_ERROR(cudaFree(gpu->dist));
}

void free_csrg_dev(csrdata *csrgraph) {
    HANDLE_ERROR(cudaFree(csrgraph->offsets));
    HANDLE_ERROR(cudaFree(csrgraph->rows));
}

void set_threads_and_blocks(int *threads, int *blocks, int *warp, int vertex) {
    int gpu, max_threads, max_blocks, sm, num_blocks, num_threads;
    cudaDeviceProp gpu_prop;

    // Leggo le proprietà del device per ottimizzare la bfs
    cudaGetDevice(&gpu);
    cudaGetDeviceProperties(&gpu_prop, gpu);

    *warp = gpu_prop.warpSize;

    max_threads = gpu_prop.maxThreadsPerBlock;
    max_blocks = gpu_prop.maxGridSize[0];
    sm = gpu_prop.multiProcessorCount;

    num_threads = *warp * 8;
    num_blocks = vertex / num_threads;
    if((vertex % num_threads) > 0) num_blocks++;
    if(num_blocks < max_blocks && num_blocks / sm > 2) {
        *blocks = num_blocks;
        *threads = num_threads;
        return;
    }
    while (num_blocks > max_blocks && num_threads < max_threads) {
        // aumento i thread
        num_threads += *warp;
        num_blocks = vertex / num_threads;
        if((vertex % num_threads) > 0) num_blocks++;
        if(num_blocks < max_blocks && num_blocks / sm > 2) {
            *blocks = num_blocks;
            *threads = num_threads;
            return;
        }
    }
    while (num_blocks / sm < 2 && num_threads > *warp) {
        // diminuisco i thread
        num_threads -= *warp;
        num_blocks = vertex / num_threads;
        if((vertex % num_threads) > 0) num_blocks++;
        if(num_blocks < max_blocks && num_blocks / sm > 2) {
            *blocks = num_blocks;
            *threads = num_threads;
            return;
        }
    }

    num_blocks = vertex / num_threads;
    if((vertex % num_threads) > 0) num_blocks++;
    *blocks = num_blocks;
    *threads = num_threads;
}
