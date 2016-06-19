#include "driverBFS.c"
#include "cuda_queue.h"
#include "lock.h"

__global__ void kernel_compute_bfs(gpudata data, csrdata csrg, Lock mutex) {
    int i, j, warp_id, increment;
    UL U, V, s, e, *node;
    /* Dimensione del warp */
    int warp_size = data.warp_size;
    /* Quanti warp ci sono in ogni blocco */
    int warps_block = blockDim.x / warp_size;
    /* Imposto l'id del Warp */
    warp_id = blockIdx.x * warps_block + threadIdx.x / warp_size;
    /* Incremento da effettuare ogni iterazione del ciclo */
    increment = (gridDim.x * blockDim.x)/warp_size;
    for(i = warp_id; i < data.nq; i+= increment) {
        U = data.queue[i];
        data.visited[U] = 1;

        s = csrg.offsets[i];
        e = csrg.offsets[i+1] - s;
        node = &csrg.rows[s];

        /* Ora faccio fare il lavoro ad ogni thread */
        for (j = threadIdx.x % warp_size; j < e; j += warp_size) {
            /* Inserisco il nodo vicino nella frontiera */
            V = node[j];
            if ((data.visited[V] != 1) && (data.dist[V] == ULONG_MAX)) {
                // Lock della risorsa
                mutex.lock();
                data.queue2[data.nq2++] = V;
                // Unlock della risorsa
                mutex.unlock();
                data.dist[V]   = data.level + 1;
            }
        }
    }
}

UL *do_bfs_cuda(UL source, csrdata *csrgraph, csrdata *csrgraph_gpu, double *cudatime, int thread_per_block)
{
    int num_threads, num_blocks, i;
    // Creo le strutture dati per i timer
    cudaEvent_t exec_start, exec_stop, alloc_copy_start, alloc_copy_stop;
    double alloc_copy_time = 0.0, bfs_time = 0.0;

    // Creo le strutture dati per passare i parametri al Device (GPU)
    gpudata host;
    gpudata dev;
    Lock mutex;

    // Inizializzo i dati prima sull' Host (CPU)
    host.queue = (UL *) Malloc(csrgraph->nv * sizeof(UL));
    host.queue2 = (UL *) Malloc(csrgraph->nv * sizeof(UL));
    host.dist = (UL *) Malloc(csrgraph->nv * sizeof(UL));
    host.visited = (char *) Malloc(csrgraph->nv * sizeof(char));
    memset(host.queue, 0, csrgraph->nv * sizeof(UL));
    memset(host.queue2, 0, csrgraph->nv * sizeof(UL));
    memset(host.visited, 0, csrgraph->nv * sizeof(char));
    for (i = 0; i < csrgraph->nv; i++) host.dist[i] = ULONG_MAX;

    host.dist[source] = 0;
    host.queue[source] = 1;
    dev.level = host.level = 0;
    dev.mutex = host.mutex = 0;
    dev.nq = host.nq = 1;
    dev.nq2 = host.nq2 = 0;
    dev.warp_size = host.warp_size = get_warp_size();

    // Inizio ad allocare memoria e copiare i dati sul device (GPU)
    START_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    copy_data_on_gpu(&host, &dev, csrgraph->nv);
    alloc_copy_time = STOP_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    printf("\nTime spent for allocation and copy: %.5f\n", alloc_copy_time);

    // Faccio partire i kernel
    START_CUDA_TIMER(&exec_start, &exec_stop);
    while(1) {
        set_threads_and_blocks(&num_threads, &num_blocks, dev.warp_size, dev.nq, thread_per_block);
        // Lancio il kernel che si occupa di mettere nella frontiera i vicini
        kernel_compute_bfs<<<num_blocks, num_threads>>>(dev, *csrgraph_gpu, mutex);
        cudaDeviceSynchronize();
        // Controllo se devo fare un ulteriore iterazione
        if(dev.nq2 == 0) {break;}
        // Inverto le code
        swap_queue(&dev);
        dev.nq = dev.nq2;
        dev.nq2 = 0;
        // Incremento la distanza
        dev.level += 1;
    }
    // Fermo il timer e stampo il tempo di esecuzione
    bfs_time = STOP_CUDA_TIMER(&exec_start, &exec_stop);
    printf("Time spent for cuda bfs: %.5f\n", bfs_time);

    // Mi copio l'array di distanze su host per effettuare il controllo
    copy_data_on_host(&host, &dev, csrgraph->nv);
    // Libero la memoria sul device
    free_gpu_data(&dev);

    // Libero la memoria sull'host
    free(host.queue);
    free(host.queue2);
    free(host.visited);
    *cudatime = alloc_copy_time + bfs_time;
    return host.dist;
}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed, int thread)
{
    csrdata csrgraph, csrgraph_gpu;     // csr data structure to represent the graph
    UL *dist;                           // array of distances from the source

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
    dist = do_bfs_cuda(root, &csrgraph, &csrgraph_gpu, &cudatime, thread);

    // Timing output
    fprintf(stdout, "\n");
    fprintf(stdout, "Cuda build csr and copy time = \t%.5f\n", csrtime);
    fprintf(stdout, "Cuda alloc data and bfs time = \t%.5f\n", cudatime);
    fprintf(stdout, "\n");

    // Libero memoria su host e device
    free_gpu_csr(&csrgraph_gpu);
    if(csrgraph.offsets) free(csrgraph.offsets);
    if(csrgraph.rows)    free(csrgraph.rows);

    return dist;
}
