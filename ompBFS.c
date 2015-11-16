#include "driverBFS.c"
#include <omp.h>

UL *do_bfs_omp(UL source, csrdata *csrg) {

    UL *q, ql, qs, i, start, end, *dist, d, U, V, s, e, j;
    int *visited;

    fprintf(stdout, "\nPerforming BFS on a graph with %lu vertices and %lu edges starting from %lu\n", csrg->nv, csrg->ne, source);
    // if(csrg->nv < 50) print_csr(csrg);

    if (csrg->deg[source] == 0) {
        fprintf(stdout, "\n\tWarning the source has degree 0\n\tNothing to do!");
        return 0;
    }
    if (source >= csrg->nv) {
        fprintf(stderr, "\nSource vertex  = %lu, not in the graph! Exit.\n\n", source);
        exit(EXIT_FAILURE);
    }

    q       = NULL;
    visited = NULL;
    dist    = NULL;

    dist      =  (UL *)Malloc(csrg->nv*sizeof(UL));
    q         =  (UL *)Malloc(csrg->ne*sizeof(UL));
    visited   = (int *)Malloc(csrg->nv*sizeof(int));

    memset(visited, 0, csrg->nv*sizeof(int));
    for (i = 0; i < csrg->nv; i++) dist[i] = ULONG_MAX;

    // enqueue the source
    q[0]         = source;
    qs           = 0;
    ql           = 1;
    d            = 0;
    dist[source] = d;
    visited[source] = 1;

    while(ql-qs !=  0) {
        start = qs; end = ql;

        #pragma omp parallel for private(U,V,s,e,j) reduction(+:qs)
        for(i = start; i < end; i++) {
            U = q[i];
            qs++;

            s = csrg->offsets[U];
            e = csrg->offsets[U+1];

            for (j = s; j < e; j++) {
                V = csrg->rows[j];
                // If V is not visited enqueue it
                if ((visited[V] != 1) && dist[V] == ULONG_MAX) {
                    if ((__sync_lock_test_and_set(&visited[V], 1)) == 0) {
                        dist[V]   = d + 1;
                        #pragma omp critical
                        {
                            q[ql++] = V;
                        }
                    }
                }
            }
		}

        fprintf(stdout, "\t Exploring level %lu,     the next queue = %lu\n", d, ql-qs);
        if(csrg->nv < 50) {
        fprintf(stdout, "\t Queue:\t");
        for (i = start; i < end; i++) fprintf(stdout, "%lu ", q[i]);
        fprintf(stdout, "\n");
        fprintf(stdout, "\t Visited:\t");
        for (i = 0; i < csrg->nv; i++) fprintf(stdout, "%d ", visited[i]);
        fprintf(stdout, "\n");
        }

        d++;
    }

    fprintf(stdout, "Finished BFS, visited %lu nodes\n", ql);
    UL count = 0;
    for (i = 0; i < csrg->nv; i++) {
        if (visited[i] == 1)
            count += 1;
    }
    if (ql != count) {
        fprintf(stderr, "\nBFS is wrong! nvisited = %lu != count = %lu.\nExit.\n\n", ql, count);
    }

    fprintf(stdout, "\n");

    free(q);
    return dist;
}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed) {

    UL *dist;             // array of distances from the source
    csrdata csrgraph;     // csr data structure to represent the graph
    FILE *fout;
    UL i;

    // Vars for timing
    struct timeval begin, end;
    double bfstime, csrtime;
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
    END_TIMER(end);
    ELAPSED_TIME(csrtime, begin, end)
    if (csrgraph.nv < 50) print_csr(&csrgraph);

    if (randsource) {
        root = random_source(&csrgraph, seed);
        fprintf(stdout, "Random source vertex %lu\n", root);
    }

    // Perform a BFS traversal that returns the array of distances from the source
    START_TIMER(begin)
    dist = do_bfs_omp(root, &csrgraph);
    END_TIMER(end);
    ELAPSED_TIME(bfstime, begin, end)

    // Print distance array to file
    fout = Fopen(DISTANCE_OUT_FILE, "w+");
    for (i = 0; i < csrgraph.nv; i++) fprintf(fout, "%lu %lu\n", i, dist[i]);
    fclose(fout);

    // Timing output
    fprintf(stdout, "\n");
    fprintf(stdout, "build csr  time = \t%.5f\n", csrtime);
    fprintf(stdout, "do BFS OMP time = \t%.5f\n", bfstime);
    fprintf(stdout, "\n");

    if(csrgraph.offsets) free(csrgraph.offsets);
    if(csrgraph.rows)    free(csrgraph.rows);

    return dist;
}
