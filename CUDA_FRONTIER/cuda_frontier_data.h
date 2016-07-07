#include "../source/cuda_error.h"
#include "../source/cuda_timer.h"
#include "../source/cuda_csr.h"

// Definisco la struttura dati che mi serviranno sul device
typedef struct _gpudata{
    char *redo;
    char *queue;
    char *frontier;
    int warp_size;
    UL *dist;
    UL vertex;
    UL level;
} gpudata;

// Copio la struttura dati apposita su device
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

// Copio i risultati sull'host
inline void copy_data_on_host(gpudata *host, gpudata *gpu) {
    HANDLE_ERROR(cudaMemcpy(host->dist, gpu->dist, (host->vertex) * sizeof(UL),cudaMemcpyDeviceToHost));
}

// Libero la memoria del device
inline void free_gpu_data(gpudata *gpu) {
    HANDLE_ERROR(cudaFree(gpu->queue));
    HANDLE_ERROR(cudaFree(gpu->frontier));
    HANDLE_ERROR(cudaFree(gpu->dist));
}

// Utility per "cercare di" scegliere un numero ottimale di thread e blocchi
void set_threads_and_blocks(int *threads, int *blocks, int *warp, int vertex, int chose_thread) {
    int gpu, max_threads, max_blocks, sm, num_blocks, num_threads;
    cudaDeviceProp gpu_prop;

    // Leggo le proprietà del device per ottimizzare la bfs
    cudaGetDevice(&gpu);
    cudaGetDeviceProperties(&gpu_prop, gpu);

    *warp = gpu_prop.warpSize;

    max_threads = gpu_prop.maxThreadsPerBlock;
    max_blocks = gpu_prop.maxGridSize[0];
    sm = gpu_prop.multiProcessorCount;

    if(chose_thread > 0 && chose_thread <= max_threads) {
        num_threads = chose_thread;
        num_blocks = vertex / num_threads;
        if((vertex % num_threads) > 0) num_blocks++;
        if(num_blocks < max_blocks) {
            *threads = num_threads;
            *blocks = num_blocks;
            return;
        }
    }

    num_threads = *warp * 8;
    num_blocks = vertex / num_threads;
    if((vertex % num_threads) > 0) num_blocks++;

    if((num_blocks <= max_blocks && chose_thread == 0) || (num_blocks <= max_blocks && num_blocks / sm > 2)) {
        *blocks = num_blocks;
        *threads = num_threads;
        return;
    }

    while (num_blocks > max_blocks && num_threads <= max_threads) {
        // aumento i thread
        num_threads += *warp;
        num_blocks = vertex / num_threads;
        if((vertex % num_threads) > 0) num_blocks++;
        if(num_blocks <= max_blocks && num_blocks / sm > 2) {
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
        if(num_blocks <= max_blocks && num_blocks / sm > 2) {
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
