#ifndef __CSRGRAPH_H__
#define __CSRGRAPH_H__

// Copio il grafo su device
inline void copy_csr_on_gpu(const csrdata *csrg, csrdata *csrg_gpu){
    UL nv = csrg->nv;
    UL ne = csrg->ne;

    HANDLE_ERROR(cudaMalloc((void**) &csrg_gpu->offsets, (nv+1) * sizeof(UL)));
    HANDLE_ERROR(cudaMalloc((void**) &csrg_gpu->rows, ne * sizeof(UL)));

    HANDLE_ERROR(cudaMemcpy(csrg_gpu->offsets, csrg->offsets, (nv+1) * sizeof(UL), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(csrg_gpu->rows, csrg->rows, ne * sizeof(UL), cudaMemcpyHostToDevice));

    csrg_gpu->nv = nv;
    csrg_gpu->ne = ne;
}

inline void free_gpu_csr(csrdata *csrgraph) {
    HANDLE_ERROR(cudaFree(csrgraph->offsets));
    HANDLE_ERROR(cudaFree(csrgraph->rows));
}

#endif
