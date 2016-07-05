#include "driverBFS.c"
#include <omp.h>

UL *do_bfs_omp(UL source, csrdata *csrg, int thread) {

    UL *q, *q2, *tmp_q, nq, nq2, i, *dist, d, U, V, s, e, j;
    char *visited;

    // Imposto il numero di thread a mano o automaticamente
    if(thread > 0 && thread < omp_get_max_threads()) {
        omp_set_num_threads(thread);
        printf("Numero Threads: %d\n", thread);
    }
    else
        printf("Numero Threads: %d\n", omp_get_max_threads());

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
    q2      = NULL;

    dist      =  (UL *)Malloc(csrg->nv*sizeof(UL));
    q         =  (UL *)Malloc(csrg->ne*sizeof(UL));
    visited   =  (char *)Malloc(csrg->nv*sizeof(char));
    q2        =  (UL *)Malloc(csrg->ne*sizeof(UL));

    memset(visited, 0, csrg->nv*sizeof(char));
    for (i = 0; i < csrg->nv; i++) dist[i] = ULONG_MAX;
    // enqueue the source
    q[0]         = source;
    nq           = 1;
    nq2          = 0;
    d            = 0;
    dist[source] = d;
    visited[source] = 1;

    // Faccio partire il ciclo per la bfs parallela
    // finchè c'è qualcosa in coda
    while(1) {
        #pragma omp parallel for private(U,V,s,e,j)
        for(i = 0; i < nq; i++) {
            U = q[i];

            s = csrg->offsets[U];
            e = csrg->offsets[U+1];

            for (j = s; j < e; j++) {
                V = csrg->rows[j];
                // If V is not visited enqueue it
                if(!visited[V] && dist[V] == ULONG_MAX) {
                    visited[V] = 1;
                    dist[V]   = d + 1;
                    #pragma omp critical
                    {
                        q2[nq2++] = V;
                    }
                }
            }
		}
        d++;
        if(nq2 == 0) {
            break;
        }
        else {
            tmp_q = q;
            q = q2;
            q2 = tmp_q;
            nq = nq2;
            nq2 = 0;
        }
    }

    free(q);
    free(visited);
    return dist;
}

UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed, int thread) {

    UL *dist;             // array of distances from the source
    csrdata csrgraph;     // csr data structure to represent the graph

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
    dist = do_bfs_omp(root, &csrgraph, thread);
    END_TIMER(end);
    ELAPSED_TIME(bfstime, begin, end)

    // Timing output
    fprintf(stdout, "\n");
    fprintf(stdout, "build csr  time = \t%.5f\n", csrtime);
    fprintf(stdout, "do BFS OMP time = \t%.5f\n", bfstime);
    fprintf(stdout, "\n");

    // Libero la memoria
    if(csrgraph.offsets) free(csrgraph.offsets);
    if(csrgraph.rows)    free(csrgraph.rows);

    return dist;
}
