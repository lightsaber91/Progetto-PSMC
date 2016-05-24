#include "driverBFS.c"
#include "cudaBFS.h"

__global__ void kernel_set_frontier(gpudata data, csrdata csrg) {

    int warp_size = data.warp_size;
    int warps_block = blockDim.x / warp_size;
    int i, j, warp_id, increment;
    UL V, s, e, *node;

    *(data.redo) = 0;
    warp_id = blockIdx.x * warps_block + threadIdx.x / warp_size;
    increment = (gridDim.x * blockDim.x)/warp_size;

    for(i = warp_id; i < csrg.nv; i+= increment) {
        if (data.queue[i]) {

            data.queue[i] = 0;
            s = csrg.offsets[i];
            e = csrg.offsets[i+1] - s;

            node = &csrg.rows[s];

            for (j = threadIdx.x % warp_size; j < e; j += warp_size) {
                V = node[j];
                data.frontier[V] = 1;
            }
        }
    }
}

__global__ void kernel_compute_distance(gpudata data) {
    UL prev_level, dist;

    int tid = (blockIdx.x*blockDim.x)+threadIdx.x;

    while (tid < data.vertex) {
        if (data.frontier[tid]) {
            *(data.redo) = 1;
            data.frontier[tid] = 0;
            prev_level = data.dist[tid];
            dist = data.level + 1;
            data.dist[tid] = (dist < prev_level) ? dist : prev_level;
            data.queue[tid] = (prev_level == ULONG_MAX);
        }
        tid += gridDim.x*blockDim.x;
    }
}

UL *do_bfs_cuda(UL source, csrdata *csrgraph, csrdata *csrgraph_gpu, double *cudatime, int thread, int counter)
{
    int num_threads, num_blocks, i;

    // Creo le strutture per i timer
    cudaEvent_t exec_start, exec_stop, alloc_copy_start, alloc_copy_stop;
    double alloc_copy_time = 0.0, bfs_time = 0.0;
    char redo = 1;

    // Dati per la gpu
    gpudata host;
    gpudata dev;

    // Leggo le proprietà del device per ottimizzare la bfs
    set_threads_and_blocks(&num_threads, &num_blocks, &(dev.warp_size), csrgraph->nv, thread);
    if(counter == 0) printf("\nNumber of threads: %d,\tNumber of blocks: %d\n", num_threads, num_blocks);

    // Inizializzo i dati
    host.level = 0;
    host.queue = (char *) Malloc(csrgraph->nv);
    host.dist = (UL *) Malloc(csrgraph->nv*sizeof(UL));
    host.vertex = csrgraph->nv;
    memset(host.queue, 0, csrgraph->nv);
    for (i = 0; i < csrgraph->nv; i++) host.dist[i] = ULONG_MAX;

    host.dist[source] = 0;
    host.queue[source] = 1;
    dev.level = host.level;

    // Inizio ad allocare memoria e copiare i dati sulla gpu
    START_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
    copy_data_on_gpu(&host, &dev);
    alloc_copy_time = STOP_CUDA_TIMER(&alloc_copy_start, &alloc_copy_stop);
//    printf("\nTime spent for allocation and copy: %.5f\n", alloc_copy_time);

    // Faccio partire il kernel
    START_CUDA_TIMER(&exec_start, &exec_stop);
    while(redo) {
        // lancio il kernel
        kernel_set_frontier<<<num_blocks, num_threads>>>(dev, *csrgraph_gpu);
        kernel_compute_distance<<<num_blocks, num_threads>>>(dev);
        dev.level += 1;
        HANDLE_ERROR(cudaMemcpy(&redo, (&dev)->redo, sizeof(char), cudaMemcpyDeviceToHost));
    }
    bfs_time = STOP_CUDA_TIMER(&exec_start, &exec_stop);
//    printf("Time spent for cuda bfs: %.5f\n", bfs_time);

    copy_data_on_host(&host, &dev);
    free_gpu_mem(&dev);
    free(host.queue);

    *cudatime = bfs_time;
    return host.dist;
}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed, int thread)
{
    csrdata csrgraph, csrgraph_gpu;     // csr data structure to represent the graph
    FILE *fout;
    UL i;
    UL *dist;             // array of distances from the source

    // Vars for timing
    struct timeval begin, end;
    double cudatime = 0.0, csrtime, tottime = 0.0;
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
//    while( thread <= 1024) {
        for(i = 0; i < 10; i++) {
          printf("%d", i);
            dist = do_bfs_cuda(root, &csrgraph, &csrgraph_gpu, &cudatime, thread, i);
            sleep(5);
            tottime += cudatime;
            free(dist);
        }
        cudatime = tottime / 10;
    // Print distance array to file
//    fout = Fopen(DISTANCE_OUT_FILE, "w+");
//    for (i = 0; i < csrgraph.nv; i++) fprintf(fout, "%lu %lu\n", i, dist[i]);
//    fclose(fout);

    // Timing output
        fprintf(stdout, "\n");
        fprintf(stdout, "Cuda build csr and copy time = \t%.5f\n", csrtime);
        fprintf(stdout, "Cuda alloc data and bfs time = \t%.5f\n", cudatime);
        fprintf(stdout, "\n");
//        if(thread == -1) thread = 32;
//        else thread *= 2;
//    }
    free_csrgraph_dev(&csrgraph_gpu);

    if(csrgraph.offsets) free(csrgraph.offsets);
    if(csrgraph.rows)    free(csrgraph.rows);

    return dist;
}
