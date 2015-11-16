#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

#define RANDMAX ((2**31)-1)
#define NBIN    (20)
#define DISTANCE_OUT_FILE "dist_file.dat"

/******************* Macros for timing **********************/
/*
    You should define the following vars
    struct timeval begin, end;
    double elapsed;
    int timer = 1;
*/

#define START_TIMER(begin)  if (timer) gettimeofday(&begin, NULL);

#define END_TIMER(end)      if (timer) gettimeofday(&end,   NULL);

//get the total number of s that the code took:
#define ELAPSED_TIME(elapsed, begin, end)            \
    if (timer) elapsed = (end.tv_sec - begin.tv_sec) \
    + ((end.tv_usec - begin.tv_usec)/1000000.0);     \

/************************************************************/

typedef unsigned long UL;

typedef struct {
    UL *rows;        // adjacency lists
    UL *offsets;     // starting points of adjacency lists
    UL *deg;         // array of degrees
    UL nv;           // number of vertices in the graph (offsets has nv+1 elements)
    UL ne;           // number of edges in the graph (elements in rows)
} csrdata;


/************************************************************/


/************************************    FUNCTIONS    **********************************************/
// Perform a bfs on a single CPU, return the array of distances from the source s
UL *do_bfs_serial(UL s, csrdata *csrg);
// Build the data struct and do bfs by calling do_bfs_serial
UL *traverse(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed);
// Validate the given distances array by calling traverse and comparing the distances
int validate_bfs(UL *edges, UL nedges, UL nvertices, UL root, UL *distances);

// parallel version of traverse
UL *traverse_parallel(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed);

// Wrong versions to check the validate function
UL *do_bfs_wrong(UL source, csrdata *csrg, int wrong);
UL *traverse_wrong(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed);

// Generate an edge list by extracting random numbers
UL *gen_graph(UL n, UL nedges, unsigned seed);
// Generate RMAT edge list
UL *gen_rmat(UL ned, int scale, float a, float ab, float abc, int seed);
// Read a graph from file
int read_graph_ff(char *fname, UL **edges, UL *nedges, UL *nvertices);

// Do some statistics in the edge list
int compute_dd(UL *edgelist, UL nedges, UL nv);
// Build the CSR data structure from the edge list
int build_csr(UL *edges, UL nedges, UL nv, csrdata *csrg);
// Make the graph undirected by mirroring edges
UL *mirror(UL *ed, UL *ned);
// remove self-loop and multiple edges
UL norm_graph(UL *ed, UL ned);
// Extract a vertex with degree > 0
UL random_source(csrdata *csrg, unsigned seed);

// Useful functions
void *Malloc(size_t sz);
static void *Realloc(void *ptr, size_t sz);
static FILE *Fopen(const char *path, const char *mode);
static int cmpedge(const void *p1, const void *p2);
int print_csr(csrdata *in);
int print_edges(UL *ed, UL ned);
int Usage (char *str);
/****************************************************************************************************/

int main(int argc, char **argv)
{

    UL nvertices, nedges;    // number of vertice, edges
    int scale, ef;           // scale (nvertices = 2^scale), average degree
    UL *edges;               // edge list
    UL root, l;              // source node for BFS
    int seed1, seed2;        // random seeds
    int randsource;          // flag to execute random source extraction
    float a, b, c, d;        // rmat parameters
    char opt;
    int errflg, gengraph, validate;
    char *fgraph_name;
    int isvalid;

    // Vars for timing
    struct timeval begin, end;
    double gentime, statstime, cleantime;
    int timer = 1;

    scale      = 4;
    ef         = 4;
    root       = 0;
    seed1      = 1;
    seed2      = 2;
    gengraph   = 1;
    errflg     = 0;
    randsource = 1;
    validate   = 0;
    isvalid    = 0;

    fgraph_name      = NULL;
    edges            = NULL;

    if (argc < 2) {
        fprintf(stdout, "\nUsing default values: scale 4, edge factor 4\n");
    } else if (argc > 9) {
        Usage(argv[0]);
    }

    while ((opt = getopt (argc, argv, "S:E:1:2:f:g:s:hV:")) != EOF){
            switch (opt)
            {
            case 'S':
                    scale = atoi(optarg);
                    break;
            case 'E':
                    ef = atof(optarg);
                    break;
            case '1':
                    seed1 = atoi(optarg);
                    break;
            case '2':
                    seed2 = atoi(optarg);
                    break;
            case 'g':
                    gengraph = atoi(optarg);
                    break;
            case 'f':
                    fgraph_name = strdup(optarg);
                    gengraph = 0;
                    break;
            case 's':
                    root = atoi(optarg);
                    randsource = 0;
                    break;
            case 'h':
                    Usage(argv[0]);
                    return 0;
//                    break;
            case ':': // option without operand
                    fprintf(stderr, "Option -%c requires an operand\n", optopt);
                    errflg++;
                    break;
            case 'V':
                    validate = atoi(optarg);
                    break;
            case '?':
                    fprintf(stderr, "Unrecognized option: -%c\n", optopt);
                    errflg++;
                    break;
            default:
                    Usage(argv[0]);
                    return 0;
        }
    }

    if (scale <= 0 || ef <= 0 || errflg > 0)
    {
        Usage(argv[0]);
        return 0;
    }

    nvertices = (1 << scale);
    nedges    = ef * nvertices;

    if ((nedges > ULONG_MAX)) {
        fprintf(stderr, "Too many edges! Exit.\n");
        exit(EXIT_FAILURE);
    }
    if (root > (nvertices - 1)) {
        fprintf(stderr, "root > (nvertices - 1)! Exit.\n");
        exit(EXIT_FAILURE);
    }
    if (gengraph == 0) {
        if (fgraph_name == NULL) {
            fprintf(stderr, "Graph file name is NULL\n");
            exit(EXIT_FAILURE);
        }
    }
    if (gengraph > 2) {
        fprintf(stderr, "Wrong value for gengraph! Must be 0, 1, 2\n");
        exit(EXIT_FAILURE);
    }

    fprintf(stdout, "\nRunning CPU BFS:\n");
    fprintf(stdout, "\tscale              = %d\n",  scale);
    fprintf(stdout, "\taverage degree     = %d\n",  ef);
    fprintf(stdout, "\tnumber of vertices = %lu\n", nvertices);
    fprintf(stdout, "\tnumber of edges    = %lu\n", nedges);
    fprintf(stdout, "\trandom seed 1      = %d\n",  seed1);
    fprintf(stdout, "\trandom seed 2      = %d\n",  seed2);
    fprintf(stdout, "\tsizeof(vertex)     = %zu\n", sizeof(nvertices));

    // Generate the graph
    a = .57;
    b = .19;
    c = .19;
    d = .05; // easy to read it's just 1 - (a+b+c)
    if(d){};
    START_TIMER(begin)
    if(gengraph == 0) read_graph_ff(fgraph_name, &edges, &nedges, &nvertices);
    if(gengraph == 1) edges = gen_graph(nvertices, nedges, (unsigned)seed1);
    if(gengraph == 2) edges = gen_rmat(nedges, scale, a, (a+b), (a+b+c), seed1);
    // print_edges(edges, nedges);
    END_TIMER(end);
    ELAPSED_TIME(gentime, begin, end)

    // make the graph undirected and remove self-loop and multi-edges
    START_TIMER(begin)
    edges = mirror(edges, &nedges);
    // Remove self-loop and multi-edges
    l = norm_graph(edges, nedges);
    END_TIMER(end);
    ELAPSED_TIME(cleantime, begin, end)
    fprintf(stdout, "The number of edges in the undirected graph is %lu\n", nedges);
    fprintf(stdout, "Removed %lu edges\n", nedges-l);
    nedges = l;

    //Degree distribution statistics
    START_TIMER(begin)
    //compute_dd(edges, nedges, nvertices);
    END_TIMER(end);
    ELAPSED_TIME(statstime, begin, end)

    // Print timing
    fprintf(stdout, "\n");
    fprintf(stdout, "generation time = \t%.5f\n", gentime);
    fprintf(stdout, "undirected time = \t%.5f\n", cleantime);
    fprintf(stdout, "do statis  time = \t%.5f\n", statstime);
    fprintf(stdout, "\n");


    UL *distances = NULL;

    if (validate == 1) {

        // YOUR GRAPH TRAVERSAL GOES HERE AND MUST RETURN THE ARRAY: UL *distances
        //////////////////////////////////////////////////////////////////////////
        //distances = traverse_wrong(edges, nedges, nvertices, root, 0, 0);
        distances = traverse_parallel(edges, nedges, nvertices, root, 0, 0);
        //////////////////////////////////////////////////////////////////////////

        isvalid = validate_bfs(edges, nedges, nvertices, root, distances);
        if (isvalid)  fprintf(stdout, "Well done!\n");
        if (!isvalid) fprintf(stdout, "There is something wrong!\n");
    }

    if (validate == 0) {
        distances = traverse(edges, nedges, nvertices, root, randsource, seed2);
    }


    if(distances)        free(distances);
    if(edges)            free(edges);
    if(fgraph_name)      free(fgraph_name);

    return 0;
}

UL *do_bfs_serial(UL source, csrdata *csrg)
{
    UL *q1, nq1;
    UL *q2, nq2;
    UL *qswap;
    UL nvisited;
    UL i, j, s, e, U, V, d;
    int *visited;
    UL *dist;

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

    q1      = NULL;
    q2      = NULL;
    visited = NULL;
    dist    = NULL;

    nq1      = 0;
    nq2      = 0;
    d        = 0;
    nvisited = 0;

    dist      =  (UL *)Malloc(csrg->nv*sizeof(UL));
    q1        =  (UL *)Malloc(csrg->ne*sizeof(UL));
    q2        =  (UL *)Malloc(csrg->ne*sizeof(UL));
    visited   = (int *)Malloc(csrg->nv*sizeof(int));

    memset(visited, 0, csrg->nv*sizeof(int));
    for (i = 0; i < csrg->nv; i++) dist[i] = ULONG_MAX;

    // enqueue the source
    q1[0]        = source;
    nq1          = 1;
    dist[source] = 0;

    // traverse the graph
    while (1) {
        fprintf(stdout, "\tExploring level %lu, elements in queue = %lu\n", d, nq1);
        for (i = 0; i < nq1; i++) {
            // dequeue U
            U = q1[i];
            // set as visited
            visited[U]  = 1;
            nvisited   += 1;
            // Search all neighbors of U
            s = csrg->offsets[U]; e = csrg->offsets[U+1];
            for (j = s; j < e; j++) {
                V = csrg->rows[j];
                // If V is not visited enqueue it
                if ((visited[V] != 1) && (dist[V] == ULONG_MAX)) {
                    if (nq2 > (csrg->ne - 1)) {fprintf(stderr, "Queue overflow error!\nExit!\n");exit(EXIT_FAILURE);}
                    q2[nq2++] = V;
                    dist[V]   = d + 1;
                }
            }
        }

        fprintf(stdout, "\tExploring level %lu,    the next queue = %lu\n", d, nq2);
        if(csrg->nv < 50) {
        fprintf(stdout, "\tcurrent queue:\t");
        for (i = 0; i < nq1; i++) fprintf(stdout, "%lu ", q1[i]);
        fprintf(stdout, "\n");
        fprintf(stdout, "\tvisited:\t");
        for (i = 0; i < csrg->nv; i++) fprintf(stdout, "%d ", visited[i]);
        fprintf(stdout, "\n");
        fprintf(stdout, "\tnext queue:\t");
        for (i = 0; i < nq2; i++) fprintf(stdout, "%lu ", q2[i]);
        fprintf(stdout, "\n");
        }
        if (nq2 == 0) break;

        nq1   = nq2;
        nq2   = 0;
        qswap = q1;
        q1    = q2;
        q2    = qswap;
        d    += 1;

        if(d > csrg->nv) {
            fprintf(stderr, "\nError: distance overflow!Exit\n\n");
            exit(EXIT_FAILURE);
        }
    }

    fprintf(stdout, "Finished BFS, visited %lu nodes\n", nvisited);
    UL count = 0;
    for (i = 0; i < csrg->nv; i++) {
        if (visited[i] == 1)
            count += 1;
    }
    if (nvisited != count) {
        fprintf(stderr, "\nBFS is wrong! nvisited = %lu != count = %lu.\nExit.\n\n", nvisited, count);
    }

    fprintf(stdout, "\n");

    return dist;
}


// Validate or do bfs: if flag = 1 VALIDATE if 0 do bfs
int validate_bfs(UL *edges, UL nedges, UL nvertices, UL root, UL *distances) {

    UL *d;           // array of distances from the source
    UL i;
    int isvalid = 1;

    d = NULL;

    d = traverse(edges, nedges, nvertices, root, 0, 0);

    for (i = 0; i < nvertices; i++) {
        if (d[i] != distances[i]) {
            fprintf(stdout, "Error the array of distances differs at element %lu\n", i);
            fprintf(stdout, "input distance = %lu, calculated distance = %lu\n", distances[i], d[i]);
            isvalid = 0;
            break;
        }
    }

    if(d) free(d);

    if (isvalid)  return 1;
    if (!isvalid) return 0;
    return 0;
}

UL *traverse(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed) {

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
    dist = do_bfs_serial(root, &csrgraph);
    END_TIMER(end);
    ELAPSED_TIME(bfstime, begin, end)

    // Print distance array to file
    fout = Fopen(DISTANCE_OUT_FILE, "w+");
    for (i = 0; i < csrgraph.nv; i++) fprintf(fout, "%lu %lu\n", i, dist[i]);
    fclose(fout);

    // Timing output
    fprintf(stdout, "\n");
    fprintf(stdout, "build csr  time = \t%.5f\n", csrtime);
    fprintf(stdout, "do BFS     time = \t%.5f\n", bfstime);
    fprintf(stdout, "\n");

    if(csrgraph.offsets) free(csrgraph.offsets);
    if(csrgraph.rows)    free(csrgraph.rows);

    return dist;
}


void *Malloc(size_t sz) {

        void *ptr;

        ptr = (void *)malloc(sz);
        if (!ptr) {
                fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
                exit(EXIT_FAILURE);
        }
        //memset(ptr, 0, sz);
        return ptr;
}

static void *Realloc(void *ptr, size_t sz) {

    void *lp;

    lp = (void *)realloc(ptr, sz);
    if (!lp && sz) {
            fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
            exit(EXIT_FAILURE);
    }
    return lp;
}

static FILE *Fopen(const char *path, const char *mode) {

    FILE *fp = NULL;
    fp = fopen(path, mode);
    if (!fp) {
            fprintf(stderr, "Cannot open file %s...\n", path);
            exit(EXIT_FAILURE);
    }
    return fp;
}

// Compare two edges used by qsort
static int cmpedge(const void *p1, const void *p2) {

    UL *l1 = (UL *)p1;
    UL *l2 = (UL *)p2;

    if (l1[0] < l2[0]) return -1;
    if (l1[0] > l2[0]) return  1;

    if (l1[1] < l2[1]) return -1;
    if (l1[1] > l2[1]) return  1;

    return 0;
}

// Simple random edges generator
UL *gen_graph(UL n, UL nedges, unsigned seed) {

    UL i;
    UL *edgelist;    // stores the list of edges as consecutive couple of vertices

    // Memory allocation for edgelist
    edgelist = (UL *)Malloc(2*nedges*sizeof(UL));

    fprintf(stdout, "\nGraph generation:\n");
    fprintf(stdout, "\tgraph type = GENERATING SIMPLE RANDOM\n");
    fprintf(stdout, "\tvertices   = %lu\n", n);
    fprintf(stdout, "\tedges      = %lu\n", nedges);

    // Init random
    srandom(seed);

    // Generate the edge list
    for (i = 0; i < nedges; i++) {
        edgelist[2*i]   = random()%n;
        edgelist[2*i+1] = random()%n;
        //fprintf(stdout, "%lu %lu\n", edgelist[2*i], edgelist[2*i+1]);
    }

    return edgelist;
}

// Generate RMAT edges
UL  *gen_rmat(UL ned, int scale, float a, float ab, float abc, int seed)
{
    UL n, i, x, y;
    UL *ed = NULL;
    int s;
    float r;

    srand48(seed);
    n = 1 << scale;
    // Memory allocation for edgelist
    ed = (UL *)Malloc(2*ned*sizeof(UL));

    fprintf(stdout, "\nGraph generation:\n");
    fprintf(stdout, "\tgraph type = GENERATING RMAT GRAPH\n");
    fprintf(stdout, "\tvertices   = %lu\n", n);
    fprintf(stdout, "\tedges      = %lu\n", ned);

    for (i = 0; i < ned; i++) {
        x = 0; y = 0;
        for(s = (1<<(scale-1)); s > 0; s >>= 1) {
                r = drand48();
                x |= s*(r > ab && r < 1.0f);
                y |= s*((r > a  && r < ab ) || (r > abc && r < 1.0f));
        }
        ed[2*i]   = x;
        ed[2*i+1] = y;
    }

    return ed;
}
/*
// Degree distribution
int compute_dd(UL *edgelist, UL nedges, UL nv)
{
    UL       *deg;         // array of degrees
    UL       Vi, Vj;       // edge ends
    float    bin;          // bin size
    int      nbin;         // number of bin
    UL       *count;       // array that stores the distribution
    int      id;           // index of count
    UL       i, totcount;

    deg = (UL *)malloc(2*nv*sizeof(UL));
    if (NULL == deg) {
        fprintf(stderr, "Error malloc deg. Exit.\n");
        exit(EXIT_FAILURE);
    }

    // Compute in degree and out degree
    for (i = 0; i < nedges; i++) {
        Vi = edgelist[2*i];
        Vj = edgelist[2*i+1];
        deg[2*Vi]   += 1;
        deg[2*Vj+1] += 1;
    }
    //for(i = 0; i < nv; i++) fprintf(stdout, "out-deg[%lu] = %lu\n", i, deg[2*i]);

    // Compute the Degree distribution
    nbin = (NBIN > nv) ? nv : NBIN;
    bin  = (float)nv/nbin;

    count = (UL *)malloc(nbin*sizeof(UL));
    if (NULL == count) {
        fprintf(stderr, "Erro malloc count! Exit.\n");
        exit(EXIT_FAILURE);
    }
    memset(count, 0, nbin*sizeof(UL));

    for (i = 0; i < nv; i++) {
        id = i/bin;
        count[id] += deg[2*i];
    }
    totcount = 0;
    for (i = 0; i < nbin; i++) {
        totcount += count[i];
    }
    if (totcount != nedges) {
        fprintf(stderr, "%s: totcount1 = %lu != nedges = %lu\n", __func__, totcount, nedges);
        exit(EXIT_FAILURE);
    }

    // Print the degree distribution
    fprintf(stdout, "\nDegree distribution:\n");
    fprintf(stdout, "\tnum vertices = %lu\n",  nv);
    fprintf(stdout, "\tnum edges    = %lu\n",  nedges);
    fprintf(stdout, "\tbin size     = %.2f\n",  bin);
    fprintf(stdout, "\tnum of bins  = %d\n", nbin);
    fprintf(stdout, "\n\t[label interval] [degree]\n");
    for (i = 0; i < nbin; i++) fprintf(stdout, "\t[%lu - %lu] = %lu\n", (UL)(i*bin), (UL)((i+1)*bin), count[i]); fflush(stdout);

    // Max out/in-going degree, zero/one-degree vertices
    UL maxind, maxoutd, maxd;
    UL zeroind, zerooutd;
    UL oneind, oneoutd;
    maxind  = maxoutd  = 0;
    zeroind = zerooutd = 0;
    oneind  = oneoutd  = 0;
    for(i = 0; i < nv; i++) {
        if(deg[2*i]   > maxoutd) maxoutd  = deg[2*i];
        if(deg[2*i+1] > maxind)  maxind   = deg[2*i+1];
        if(deg[2*i]   == 0)      zerooutd += 1;
        if(deg[2*i+1] == 0)      zeroind  += 1;
        if(deg[2*i]   == 1)      oneoutd  += 1;
        if(deg[2*i+1] == 1)      oneind   += 1;
    }
    maxd = (maxind > maxoutd) ? maxind : maxoutd;

    // Count how many nodes has a certain value of out-degree
    UL minnbin = maxoutd + 1;
    nbin = (NBIN > minnbin) ? minnbin : NBIN;
    bin  = (float)minnbin/nbin;

    UL *newcount;
    newcount = (UL *)malloc(nbin*sizeof(UL));
    if (NULL == newcount) {
        fprintf(stderr, "Erro malloc count! Exit.\n");
        exit(EXIT_FAILURE);
    }
    memset(newcount, 0, nbin*sizeof(UL));

    for (i = 0; i < nv; i++) {
        id = deg[2*i]/bin;
        newcount[id] += 1;
    }
    totcount = 0;
    for (i = 0; i < nbin; i++) {
        totcount += newcount[i];
    }

    fprintf(stdout, "\nNode-Degree distribution:\n");
    fprintf(stdout, "\tnum vertices = %lu\n",  nv);
    fprintf(stdout, "\tnum edges    = %lu\n",  nedges);
    fprintf(stdout, "\tbin size     = %.2f\n", bin);
    fprintf(stdout, "\tnum of bins  = %d\n",   nbin);
    fprintf(stdout, "\tmax in-deg   = %lu\n",  maxind);
    fprintf(stdout, "\tmax out-eg   = %lu\n",  maxoutd);
    fprintf(stdout, "\tzero in-deg  = %lu\n",  zeroind);
    fprintf(stdout, "\tzero out-deg = %lu\n",  zerooutd);
    fprintf(stdout, "\tone in-deg   = %lu\n",  oneind);
    fprintf(stdout, "\tone out-deg  = %lu\n",  oneoutd);
    fprintf(stdout, "\n\t[degree interval] [nodes]\n");
    for (i = 0; i < nbin; i++) fprintf(stdout, "\t[%lu - %lu] = [%lu]\n", (UL)(i*bin), (UL)((i+1)*bin), newcount[i]); fflush(stdout);
    if (totcount != nv) {
        fprintf(stderr, "%s: totcount2 = %lu != nedges = %lu\n", __func__, totcount, nv);
        exit(EXIT_FAILURE);
    }

    if(deg)      free(deg);
    if(count)    free(count);
    if(newcount) free(newcount);

    return 0;
}
*/
/*
    Read a directed graph from file
    File format:
    #
    # Any number of lines starting with #
    #
    # Nodes: 12345 Edges: 123456
    #
    # Any number of lines starting with #
    #
    V0  V1
    V2  V3
    ...
*/
#define BUFFSIZE     1024
#define ALLOC_BLOCK (2*1024)
int read_graph_ff(char *fname, UL **edges, UL *nedges, UL *nvertices)
{
    FILE *fp;
    char buffer[BUFFSIZE];
    int buflen;
    int comment_count;
    UL nv, ne, i, j;
    UL line_count, n;
    UL nmax;
    UL *ed;

    nv = 0;
    ne = 0;

    fp = Fopen(fname, "r");

    comment_count = 0;
    line_count    = 0;
    n             = 0;
    nmax          = ALLOC_BLOCK; // must be even

    ed = NULL;
    ed = (UL *)Malloc(nmax*sizeof(UL));

    fprintf(stdout, "\nReading graph from file\n");

    while(1) {

        // READ LINES
        fgets(buffer, BUFFSIZE, fp);
        buflen = strlen(buffer);
        if (buflen >= BUFFSIZE) {
            fprintf(stderr, "The line is to long, increase the BUFFSIZE! Exit\n");
            exit(EXIT_FAILURE);
        }

        if (feof(fp)) {
            fprintf(stdout, "\tSuccessfully read %lu lines of %s.\n", n/2, fname);
            fclose(fp);
            break;
        } else if (ferror(fp)) {
            fprintf(stderr, "\nAn error ocurred while reading the file\n");
            fclose(fp);
            perror("MAIN:");
        }

        // SCAN THE LINE
       if (strstr(buffer, "#") != NULL) {
            //fprintf(stdout, "\nreading line number %lu: %s\n", line_count, buffer);
            if (strstr(buffer, "Nodes:")) {
                        sscanf(buffer, "# Nodes: %lu Edges: %lu\n", &nv, &ne);
                        fprintf(stdout, "\tnv = %lu ne = %lu\n", nv, ne);
                }
                comment_count++;
        } else {
                //fprintf(stdout, "\nreading line number %lu: %s\n", line_count, buffer);
                if ((nv == 0) || (ne == 0)){
                    fprintf(stderr, "Error reading the number of vertices or edges in %s\n", fname);
                    exit(EXIT_FAILURE);
                }

                // Read edges
                sscanf(buffer, "%lu %lu\n", &i, &j);

            if (i >= nv || j >= nv) {
                fprintf(stderr, "found invalid edge in %s, line %lu, edge: (%lu, %lu)\n",
                fname, (comment_count + line_count), i, j);
                exit(EXIT_FAILURE);
                }

                if (n >= nmax) {
                    nmax += ALLOC_BLOCK;
                    ed   = (UL *)Realloc(ed, nmax*sizeof(UL));
                }

                ed[n]   = i;
                ed[n+1] = j;
                n       += 2;
                //fprintf(stdout, "\treading line number %lu: %lu %lu\n", n/2, ed[n/2], ed[n/2+1]);
        }
        line_count++;

    }

    fprintf(stdout, "\n");

    /*
    if ((comment_count > 0) && (line_count != (ne + comment_count))) {
        fprintf(stderr, "Error reading the input file %s: the number of lines differ from the number of edges in the header\n", fname);
        fprintf(stderr, "lcounter = %lu nedges + comm = %lu\n", line_count, (ne + comment_count));
        exit(EXIT_FAILURE);
    }
    */

    if (ne != n/2) {
        fprintf(stderr, "Error reading the input file %s: the number of edges read differ from the number of edges int the header\n", fname);
        fprintf(stdout, "nedges header = %lu edges lines read = %lu\n", ne, n/2);
        exit(EXIT_FAILURE);
    }


    *nedges    = ne;
    *nvertices = nv;
    *edges     = ed;

    n = n/2;

    return n;
}
#undef BUFFSIZE

int build_csr(UL *edges, UL nedges, UL nv, csrdata *csrg)
{

    UL i, Vi;

    csrg->nv = nv;
    csrg->ne = nedges;

    memset(csrg->deg, 0, (nv*sizeof(UL)));

    // Compute out degree
    for (i = 0; i < nedges; i++) {
        Vi = edges[2*i];
        csrg->deg[Vi] += 1;
    }
    //for (i = 0; i < nv; i++)   fprintf(stdout, "outdegree[%lu] = %lu\n",  i, csrg->deg[i]);

    // Exclusive scan on the (out-)degree array -> offset array of the CSR
    csrg->offsets[0] = 0;
    for (i = 1; i <= nv; i++) {
        csrg->offsets[i] = csrg->offsets[i-1] + csrg->deg[i-1];
        // fprintf(stdout, "csr->offsets[%lu] = %lu\n", i, csr->offsets[i]);
    }

    if (csrg->offsets[nv] != nedges) {
        fprintf(stderr, "\nError computing cumulative degrees array! Exit\n");
        exit(EXIT_FAILURE);
    }

    // Build rows array
    qsort(edges, nedges, sizeof(UL[2]), cmpedge);
    // for (i = 0; i < nedges; i++) fprintf(stdout, "%lu %lu\n", edges[2*i], edges[2*i+1]);

    for (i = 0; i < nedges; i++) {
        csrg->rows[i] = edges[2*i + 1];
    }

    return 0;
}


// Mirror edges with the exception of self loops. This function
// is used to make the graph undirected
UL *mirror(UL *ed, UL *ned) {

        UL i, n;

        ed = (UL *)Realloc(ed, (ned[0]*(2+2))*sizeof(UL));

        n = 0;
        for(i = 0; i < ned[0]; i++) {
                if (ed[2*i] != ed[2*i+1]) {
                        ed[2*ned[0]+2*n] = ed[2*i+1];
                        ed[2*ned[0]+2*n+1] = ed[2*i];
                        n++;
                } else {
                    //fprintf(stdout, "SELF LOOP FOUND, SKIP\n");
                }
        }
        ned[0] += n;

        fprintf(stdout, "\tMirroring edges -> Graph will be undirected, nedges = %lu\n", ned[0]);

        return ed;
}

// Remove self loops and multi-edges from the edge list
UL norm_graph(UL *ed, UL ned) {

    UL l, n;

    if (ned == 0) return 0;

    fprintf(stdout, "\tRemoving self loop and multi edges\n");
    printf("N egdes %lu\n\n\n\n", ned);
/*    for(n = 0; n < ned; n++) printf("%lu - ", ed[n]);
    printf("\n\n");

    parallel_qs(ed, 0, 2*ned-2, 1);

    for(n = 0; n < ned; n++) printf("%lu - ", ed[n]);
    printf("\n\n");*/

    qsort(ed, ned, sizeof(UL[2]), cmpedge);

    //for(n = 0; n < ned; n++) printf("%lu - ", ed[n]);

    // Handle first self-loop
    l = (ed[0] == ed[1]) ? 0 : 1;

    for(n = 1; n < ned; n++) {
            if ( ((ed[2*n] != ed[2*(n-1)])  || (ed[2*n+1] != ed[2*(n-1)+1])) && (ed[2*n] != ed[2*n+1]) )
            {
                ed[2*l]   = ed[2*n];
                ed[2*l+1] = ed[2*n+1];
                l++;
            } else {
                //printf("removing (%lu,%lu) , (%lu,%lu)\n", ed[2*(n-1)], ed[2*(n-1)+1], ed[2*n], ed[2*n+1]);
            }
    }
    fprintf(stdout, "\tEdges after removing = %lu\n", l);

    return l;
}

int print_csr(csrdata *in) {

    UL i, j, s, e;

    fprintf(stdout, "\nCSR data structure:\n");
    for (i = 0; i < in->nv; i++) {
        fprintf(stdout, "\t%lu: %lu:", i, in->deg[i]);
        s = in->offsets[i];
        e = in->offsets[i+1];
        for (j = s; j < e; j++) {
            fprintf(stdout, "%lu ", in->rows[j]);
        }
        fprintf(stdout, "\n");
    }
    return 0;
}

int print_edges(UL *ed, UL ned)
{
    UL i;
    for (i = 0; i < ned; i++) {
        fprintf(stdout, "\t%lu %lu\n", ed[2*i], ed[2*i+1]);
    }
    return 0;
}

// Extract a random vertex with degree > 0
UL random_source(csrdata *csrg, unsigned seed)
{
    UL s;

    // Init random
    srandom(seed);

    while (1) {
        s = random()%(csrg->nv);
        if (csrg->deg[s] > 0) break;
    }

    if (s > (csrg->nv - 1)) {
        fprintf(stderr, "source vertex = %lu. Impossible! Exit\n", s);
        exit(EXIT_FAILURE);
    }

    return s;
}

int Usage (char *str) {
    fprintf(stderr, "usage: %s [-S scale] [-E average degree] [-1 seed1] [-2 seed2] [-s source] [-g 0, 1, 2] [-f input graph file] [-h usage] [-V validate]\n", str);
    fprintf(stderr, "without parameters runs with default values:\n"
    "scale      = 4\n"
    "ef         = 4\n"
    "source     = 0\n"
    "seed1      = 1\n"
    "seed2      = 2\n"
    "validate   = 0\n"
    "gengraph   = 1   0 -> read from file, needs -f filename; 1 -> simple random; 2 -> rmat \n");

    return 0;
}
/*
UL *traverse_wrong(UL *edges, UL nedges, UL nvertices, UL root, int randsource, int seed) {

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

    if (randsource) {
        root = random_source(&csrgraph, seed);
        fprintf(stdout, "Random source vertex %lu\n", root);
    }

    // Perform a BFS traversal that returns the array of distances from the source
    START_TIMER(begin)
    dist = do_bfs_wrong(root, &csrgraph, 0); // if wrong != 0 who knows?
    END_TIMER(end);
    ELAPSED_TIME(bfstime, begin, end)

    // Print distance array to file
    fout = Fopen(DISTANCE_OUT_FILE, "w+");
    for (i = 0; i < csrgraph.nv; i++) fprintf(fout, "%lu %lu\n", i, dist[i]);
    fclose(fout);

    // Timing output
    fprintf(stdout, "\n");
    fprintf(stdout, "build csr  time = \t%.5f\n", csrtime);
    fprintf(stdout, "do BFS     time = \t%.5f\n", bfstime);
    fprintf(stdout, "\n");

    if(csrgraph.offsets) free(csrgraph.offsets);
    if(csrgraph.rows)    free(csrgraph.rows);

    return dist;
}

UL *do_bfs_wrong(UL source, csrdata *csrg, int wrong)
{
    UL *q1, nq1;
    UL *q2, nq2;
    UL *qswap;
    UL nvisited;
    UL i, j, s, e, U, V, d;
    int *visited;
    UL *dist;

    q1      = NULL;
    q2      = NULL;
    visited = NULL;
    dist    = NULL;

    nq1      = 0;
    nq2      = 0;
    d        = 0;
    nvisited = 0;

    dist      =  (UL *)Malloc(csrg->nv*sizeof(UL));
    q1        =  (UL *)Malloc(csrg->ne*sizeof(UL));
    q2        =  (UL *)Malloc(csrg->ne*sizeof(UL));
    visited   = (int *)Malloc(csrg->nv*sizeof(int));

    memset(visited, 0, csrg->nv*sizeof(int));
    for (i = 0; i < csrg->nv; i++) dist[i] = ULONG_MAX;

    // enqueue the source
    q1[0]        = source;
    nq1          = 1;
    dist[source] = 0;

    // traverse the graph
    while (1) {
        for (i = 0; i < nq1; i++) {
            // dequeue U
            U = q1[i];
            // set as visited
            visited[U]  = 1;
            nvisited   += 1;
            // Search all neighbors of U
            s = csrg->offsets[U]; e = csrg->offsets[U+1];
            for (j = s; j < e; j++) {
                V = csrg->rows[j];
                // If V is not visited enqueue it
                if ((visited[V] != 1) && (dist[V] == ULONG_MAX)) {
                    if (nq2 > (csrg->ne - 1)) {fprintf(stderr, "Queue overflow error!\nExit!\n");exit(EXIT_FAILURE);}
                    q2[nq2++] = V;
                    dist[V]   = d + 1;
                }
            }
        }

        if (nq2 == 0) break;

        // WRONGNESS ////////////
        if (d == wrong) q2[nq2-1] = 0;
        /////////////////////////
        nq1   = nq2;
        nq2   = 0;
        qswap = q1;
        q1    = q2;
        q2    = qswap;
        d    += 1;

        if(d > csrg->nv) {
            fprintf(stderr, "\nError: distance overflow!Exit\n\n");
            exit(EXIT_FAILURE);
        }
    }

    free(q1);
    free(q2);
    free(visited);

    return dist;
}
*/
