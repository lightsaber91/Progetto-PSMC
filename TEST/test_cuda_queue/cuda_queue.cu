#include "driverBFS.c"
#include "cuda_queue.h"

__global__ void kernel_compute_bfs(gpudata data, csrdata csrg) {

    unsigned int i, j, warp_id, increment, my_location;
    UL U, V, s, e, *node;
    /* Dimensione del warp */
    int warp_size = data.warp_size;
    /* Quanti warp ci sono in ogni blocco */
    int warps_block = blockDim.x / warp_size;
    /* Imposto l'id del Warp */
    warp_id = blockIdx.x * warps_block + threadIdx.x / warp_size;
    /* Incremento da effettuare ogni iterazione del ciclo */
    increment = (gridDim.x * blockDim.x)/warp_size;

    for(i = warp_id; i < *(data.nq); i+= increment) {
        U = data.queue[i];

        s = csrg.offsets[U];
        e = csrg.offsets[U+1] - s;
        node = &csrg.rows[s];

        /* Ora faccio fare il lavoro ad ogni thread */
        for (j = threadIdx.x % warp_size; j < e; j += warp_size) {
            /* Inserisco il nodo vicino nella frontiera */
            V = node[j];
            if ((data.visited[V] != 1) && (data.dist[V] == ULONG_MAX)) {
				data.visited[V] = 1;
				my_location = atomicAdd((unsigned int *) data.nq2, (unsigned int) 1);
                data.queue2[my_location] = V;
                data.dist[V] = data.level + 1;
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

    // Inizializzo i dati prima sull' Host (CPU)
    host.queue = (int *) Malloc(csrgraph->nv * sizeof(int));
    host.queue2 = (int *) Malloc(csrgraph->nv * sizeof(int));
    host.dist = (UL *) Malloc(csrgraph->nv * sizeof(UL));
    host.visited = (char *) Malloc(csrgraph->nv * sizeof(char));
    host.nq = (int *) Malloc(sizeof(int));
    host.nq2 = (int *) Malloc(sizeof(int));
    *(host.nq) = 1;
    *(host.nq2) = 0;
    memset(host.queue, 0, csrgraph->nv * sizeof(int));
    memset(host.queue2, 0, csrgraph->nv * sizeof(int));
    memset(host.visited, 0, csrgraph->nv * sizeof(char));
    for (i = 0; i < csrgraph->nv; i++) host.dist[i] = ULONG_MAX;

    host.dist[source] = 0;
    host.visited[source] = 1;
    host.queue[0] = source;
    dev.level = host.level = 0;
    dev.warp_size = host.warp_size = get_warp_size();

    // Inizio ad allocare memoria e copiare i dati sul device (GPU)
    START_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    copy_data_on_gpu(&host, &dev, csrgraph->nv);
    alloc_copy_time = STOP_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
//    printf("\nTime spent for allocation and copy: %.5f\n", alloc_copy_time);
    // Faccio partire i kernel
    START_CUDA_TIMER(&exec_start, &exec_stop);
    while(1) {
        set_threads_and_blocks(&num_threads, &num_blocks, dev.warp_size, *(host.nq), thread_per_block);
        // Lancio il kernel che si occupa di mettere nella frontiera i vicini
        kernel_compute_bfs<<<num_blocks, num_threads>>>(dev, *csrgraph_gpu);
        // Controllo e inverto le code
        if(swap_queues_and_check(&host, &dev, csrgraph->nv) == false) {
            break;
        }
        // Incremento la distanza
        dev.level += 1;
    }
    // Fermo il timer e stampo il tempo di esecuzione
    bfs_time = STOP_CUDA_TIMER(&exec_start, &exec_stop);
//    printf("Time spent for cuda bfs: %.5f\n", bfs_time);

    // Mi copio l'array di distanze su host per effettuare il controllo
    copy_data_on_host(&host, &dev, csrgraph->nv);
    // Libero la memoria sul device
    free_gpu_data(&dev);

    // Libero la memoria sull'host
    free(host.queue);
    free(host.queue2);
    free(host.visited);
    *cudatime = bfs_time;
    return host.dist;
}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed, int thread)
{
    csrdata csrgraph, csrgraph_gpu;     // csr data structure to represent the graph
    UL *dist;                           // array of distances from the source

    // Vars for timing
    struct timeval begin, end;
    double cudatime = 0.0, csrtime, total = 0.0;
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
    for(int i = 0; i < 10; i++) {
        dist = do_bfs_cuda(root, &csrgraph, &csrgraph_gpu, &cudatime, thread);
        if(i < 9) free(dist);
        total += cudatime;
    }
    cudatime = total / 10;
    // Timing output
    fprintf(stdout, "\n");
    fprintf(stdout, "Cuda build csr= \t%.5f\n", csrtime);
    fprintf(stdout, "Cuda bfs time = \t%.5f with:= %d thread per block\n", cudatime, thread);
    fprintf(stdout, "\n");

    // Libero memoria su host e device
    free_gpu_csr(&csrgraph_gpu);
    if(csrgraph.offsets) free(csrgraph.offsets);
    if(csrgraph.rows)    free(csrgraph.rows);

    return dist;
}
