#include "driverBFS.c"
#include "cudaBFS.h"

__global__ void kernel_set_frontier(gpudata data, csrdata csrg) {

    int i, j, warp_id, increment;
    UL V, s, e, *node;
    /* Dimensione del warp */
    int warp_size = data.warp_size;
    /* Quanti warp ci sono in ogni blocco */
    int warps_block = blockDim.x / warp_size;

    /* Dico di non fare altre iterazioni */
    *(data.redo) = 0;
    /* Imposto l'id del Warp */
    warp_id = blockIdx.x * warps_block + threadIdx.x / warp_size;
    /* Incremento da effettuare ogni iterazione del ciclo */
    increment = (gridDim.x * blockDim.x)/warp_size;

    for(i = warp_id; i < csrg.nv; i+= increment) {
        /* Controllo se l' i-esimo nodo è in coda */
        if (data.queue[i]) {

            /* Lo rimuovo dalla coda e mi prendo tutti i suoi vicini */
            data.queue[i] = 0;
            s = csrg.offsets[i];
            e = csrg.offsets[i+1] - s;
            node = &csrg.rows[s];

            /* Ora faccio fare il lavoro ad ogni thread */
            for (j = threadIdx.x % warp_size; j < e; j += warp_size) {
                /* Inserisco il nodo vicino nella frontiera */
                V = node[j];
                data.frontier[V] = 1;
            }
        }
    }
}

__global__ void kernel_compute_distance(gpudata data) {
    UL prev_level, dist;
    /* Imposto l'id per ogni thread */
    UL tid = (blockIdx.x*blockDim.x)+threadIdx.x;

    /* Controllo che l'id non sia maggiore del numero di nodi */
    while (tid < data.vertex) {
        /* Se il nodo con indice tid è presente nella frontiera */
        if (data.frontier[tid]) {
            /* Devo fare almeno un'altra iterazione */
            *(data.redo) = 1;
            /* Lo rimuovo dalla frontiera */
            data.frontier[tid] = 0;
            /* Faccio un controllo per vedere se già visitato */
            prev_level = data.dist[tid];
            dist = data.level + 1;
            data.dist[tid] = (dist < prev_level) ? dist : prev_level;
            /* Se non era stato visitato lo metto in coda */
            data.queue[tid] = (prev_level == ULONG_MAX);
        }
        /* Incremento l'id del thread */
        tid += gridDim.x*blockDim.x;
    }
}

UL *do_bfs_cuda(UL source, csrdata *csrgraph, csrdata *csrgraph_gpu, double *cudatime, int thread)
{
    int num_threads, num_blocks, i;

    // Creo le strutture dati per i timer
    cudaEvent_t exec_start, exec_stop, alloc_copy_start, alloc_copy_stop;
    double alloc_copy_time = 0.0, bfs_time = 0.0;
    char redo = 1;

    // Creo le strutture dati per passare i parametri al Device (GPU)
    gpudata host;
    gpudata dev;

    // Leggo le proprietà del device per ottimizzare la bfs
    set_threads_and_blocks(&num_threads, &num_blocks, &(dev.warp_size), csrgraph->nv, thread);
    printf("\nNumber of threads: %d,\tNumber of blocks: %d\n", num_threads, num_blocks);

    // Inizializzo i dati prima sull' Host (CPU)
    host.level = 0;
    host.queue = (char *) Malloc(csrgraph->nv);
    host.dist = (UL *) Malloc(csrgraph->nv*sizeof(UL));
    host.vertex = csrgraph->nv;
    memset(host.queue, 0, csrgraph->nv);
    for (i = 0; i < csrgraph->nv; i++) host.dist[i] = ULONG_MAX;

    host.dist[source] = 0;
    host.queue[source] = 1;
    dev.level = host.level;

    // Inizio ad allocare memoria e copiare i dati sul device (GPU)
    START_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    copy_data_on_gpu(&host, &dev);
    alloc_copy_time = STOP_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    printf("\nTime spent for allocation and copy: %.5f\n", alloc_copy_time);

    // Faccio partire i kernel
    START_CUDA_TIMER(&exec_start, &exec_stop);
    while(redo) {
        // lancio il kernel che si occupa di mettere nella frontiera i vicini
        kernel_set_frontier<<<num_blocks, num_threads>>>(dev, *csrgraph_gpu);
        // Lancio il kernel che si occupa di calcolare le distanze
        kernel_compute_distance<<<num_blocks, num_threads>>>(dev);
        // Incremento la distanza
        dev.level += 1;
        // Controllo se devo fare un ulteriore iterazione
        HANDLE_ERROR(cudaMemcpy(&redo, (&dev)->redo, sizeof(char), cudaMemcpyDeviceToHost));
    }
    // Fermo il timer e stampo il tempo di esecuzione
    bfs_time = STOP_CUDA_TIMER(&exec_start, &exec_stop);
    printf("Time spent for cuda bfs: %.5f\n", bfs_time);

    // Mi copio l'array di distanze su host per effettuare il controllo
    copy_data_on_host(&host, &dev);
    // Libero la memoria sul device
    free_gpu_data(&dev);

    // Libero la memoria sull'host
    free(host.queue);
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
