// Definisco la struttura dati che mi serviranno sul device
typedef struct _gpudata{
    int warp_size;
    int *visited;
    UL *dist;
    int *queue;
    int *queue2;
    int *nq;
    int *nq2;
    UL level;
} gpudata;


// Funzione di controllo per vedere se si va in errore sul device
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static inline void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}

// Funzione per far partire il timer
static inline void START_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop)
{
	HANDLE_ERROR(cudaEventCreate(start));
	HANDLE_ERROR(cudaEventCreate(stop));
	HANDLE_ERROR(cudaEventRecord(*start, 0));
}
// Funzione per fermare il timer e calcolare il tempo trascorso
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

// Copio il grafo su device
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

// Copio la struttura dati apposita su device
inline void copy_data_on_gpu(const gpudata *host, gpudata *gpu, UL vertex) {

    HANDLE_ERROR(cudaMalloc((void**) &gpu->queue, vertex * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->queue2, vertex * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->dist, vertex * sizeof(UL)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->visited, vertex * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->nq, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->nq2, sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(gpu->queue, host->queue, vertex * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->queue2, host->queue2, vertex * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->dist, host->dist, vertex * sizeof(UL), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->visited, host->visited, vertex * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->nq, host->nq, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->nq2, host->nq2, sizeof(int), cudaMemcpyHostToDevice));
}

// Copio i risultati sull'host
inline void copy_data_on_host(gpudata *host, gpudata *gpu, UL vertex) {
    HANDLE_ERROR(cudaMemcpy(host->dist, gpu->dist, vertex * sizeof(UL),cudaMemcpyDeviceToHost));
}

// Libero la memoria del device
inline void free_gpu_data(gpudata *gpu) {
    HANDLE_ERROR(cudaFree(gpu->nq));
    HANDLE_ERROR(cudaFree(gpu->nq2));
    HANDLE_ERROR(cudaFree(gpu->queue));
    HANDLE_ERROR(cudaFree(gpu->queue2));
    HANDLE_ERROR(cudaFree(gpu->dist));
    HANDLE_ERROR(cudaFree(gpu->visited));
}

inline void free_gpu_csr(csrdata *csrgraph) {
    HANDLE_ERROR(cudaFree(csrgraph->offsets));
    HANDLE_ERROR(cudaFree(csrgraph->rows));
}

inline bool swap_queues_and_check(gpudata *host, gpudata *gpu, UL vertex) {
    int nq = 0;
    HANDLE_ERROR(cudaMemcpy(host->nq2, gpu->nq2, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host->queue2, gpu->queue2, vertex * sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMemcpy(gpu->nq, host->nq2, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->queue, host->queue2, vertex * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->nq2, &nq, sizeof(int), cudaMemcpyHostToDevice));

    if(*(host->nq2) == 0) {
        return false;
    }
    *(host->nq) = *(host->nq2);
    return true;
}

inline int get_warp_size() {
  int gpu;
  cudaDeviceProp gpu_prop;

  // Leggo le proprietà del device per ottimizzare la bfs
  cudaGetDevice(&gpu);
  cudaGetDeviceProperties(&gpu_prop, gpu);

  return gpu_prop.warpSize;
}

// Utility per "cercare di" scegliere un numero ottimale di thread e blocchi
inline void set_threads_and_blocks(int *threads, int *blocks, int warp, UL queue_len, int thread_per_block) {
    int gpu, max_threads, max_blocks, num_blocks, num_threads, max_thread_per_block = 256;
    cudaDeviceProp gpu_prop;

    cudaGetDevice(&gpu);
    cudaGetDeviceProperties(&gpu_prop, gpu);

    max_threads = gpu_prop.maxThreadsPerBlock;
    max_blocks = gpu_prop.maxGridSize[0];

    num_threads = queue_len * warp;

    if(thread_per_block > 0 && thread_per_block <= max_threads) {
        num_blocks = num_threads / thread_per_block;
        if(num_threads % thread_per_block > 0) num_blocks++;
        if(num_blocks <= max_blocks) {
            *threads = thread_per_block;
            *blocks = num_blocks;
            return;
        }
        do {
            num_blocks /= 2;
        } while(num_blocks > max_blocks);
        *threads = thread_per_block;
        *blocks = num_blocks;
        return;
    }

    num_blocks = num_threads / max_thread_per_block;
    if(num_threads % max_thread_per_block > 0) num_blocks++;
    if(num_blocks <= max_blocks) {
        *threads = max_thread_per_block;
        *blocks = num_blocks;
        return;
    }
    do {
        num_blocks /= 2;
    } while(num_blocks > max_blocks);
    *threads = max_thread_per_block;
    *blocks = num_blocks;
    return;
}
