#include <cuda.h>

#define WARP 32

typedef struct {
    char *redo;
    char *queue;
    char *next_queue;
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
	return time;
}


void copy_csr_on_gpu(const csrdata *csrg, csrdata *csrg_gpu){
    UL nv = csrg->nv;
    UL ne = csrg->ne;

    HANDLE_ERROR(cudaMalloc((void**) &csrg_gpu->offsets, (nv+1) * sizeof(UL)));
    HANDLE_ERROR(cudaMalloc((void**) &csrg_gpu->rows, ne * sizeof(UL)));

    HANDLE_ERROR(cudaMemcpy(csrg_gpu->offsets, csrg->offsets, (nv+1) * sizeof(UL), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(csrg_gpu->rows, csrg->rows, ne * sizeof(UL), cudaMemcpyHostToDevice));

    csrg_gpu->nv = nv;
    csrg_gpu->ne = ne;
}

void copy_data_on_gpu(const gpudata *host, gpudata *gpu) {
    gpu->vertex = host->vertex;

    HANDLE_ERROR(cudaMalloc((void**) &gpu->redo, sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->queue, host->vertex * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->next_queue, host->vertex * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->dist, host->vertex * sizeof(UL)));

    HANDLE_ERROR(cudaMemcpy(gpu->queue, host->queue, host->vertex * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->dist, host->dist, host->vertex * sizeof(char), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemset(gpu->next_queue, 0, host->vertex * sizeof(char)));
}

void copy_data_on_host(gpudata *host, gpudata *gpu) {
    HANDLE_ERROR(cudaMemcpy(host->dist, gpu->dist, (host->vertex) * sizeof(char),cudaMemcpyDeviceToHost));
    free(host->queue);
    free(host->next_queue);
}

void free_gpu_mem(gpudata *gpu, csrdata *csrgraph) {
    HANDLE_ERROR(cudaFree(gpu->queue));
    HANDLE_ERROR(cudaFree(gpu->next_queue));
    HANDLE_ERROR(cudaFree(gpu->dist));
    HANDLE_ERROR(cudaFree(csrgraph->offsets));
    HANDLE_ERROR(cudaFree(csrgraph->rows));
}
