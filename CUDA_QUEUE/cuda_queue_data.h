#include "../source/cuda_error.h"
#include "../source/cuda_timer.h"
#include "../source/cuda_csr.h"
#include "cuda_queue_utils.h"

// Definisco la struttura dati che mi serviranno sul device
typedef struct _gpudata{
    UL level;                   // Livello del grafo esplorato
    UL *dist;                   // Array per salvare le distanze
    char *visited;              // Array per impostare i nodi come visitati
    int warp_size;              // Dimensione del Warp
    int *queue;                 // Coda da analizzare in questo livello
    int *queue2;                // Coda da analizzare nell'iterazione successiva
    int *nq;                    // Lunghezza della coda attuale
    int *nq2;                   // Lunghezza della seconda coda
    int *offset;                // Da quale vicino il warp deve partire
} gpudata;

// Copio la struttura dati apposita su device
inline void copy_data_on_gpu(const gpudata *host, gpudata *gpu, UL vertex, UL edges) {

    HANDLE_ERROR(cudaMalloc((void**) &gpu->queue, edges * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->queue2, edges * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->dist, vertex * sizeof(UL)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->visited, vertex * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->nq, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->nq2, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &gpu->offset, edges * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(gpu->queue, host->queue, edges * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->queue2, host->queue2, edges * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->dist, host->dist, vertex * sizeof(UL), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->visited, host->visited, vertex * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->nq, host->nq, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->nq2, host->nq2, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu->offset, host->offset, edges * sizeof(int), cudaMemcpyHostToDevice));
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
    HANDLE_ERROR(cudaFree(gpu->offset));
}

inline bool swap_queues_and_check(gpudata *host, gpudata *gpu, UL vertex) {
    int zero = 0, *temp;
    // Effettuo il check su nq2 --> Se maggiore di zero devo itarare nuovamente
    HANDLE_ERROR(cudaMemcpy(host->nq2, gpu->nq2, sizeof(int), cudaMemcpyDeviceToHost));
    if(*(host->nq2) == 0) {
        return false;
    }
    // Procedo ad invertire le code
    temp = gpu->queue;
    gpu->queue = gpu->queue2;
    gpu->queue2 = temp;
    // Inverto anche i contatori e resetto nq2
    temp = gpu->nq;
    gpu->nq = gpu->nq2;
    gpu->nq2 = temp;
    HANDLE_ERROR(cudaMemcpy(gpu->nq2, &zero, sizeof(int), cudaMemcpyHostToDevice));
    // Inverto anche su cpu
    host->nq = host->nq2;
    return true;
}
