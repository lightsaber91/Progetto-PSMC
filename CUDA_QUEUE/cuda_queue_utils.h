#ifndef __UTILS_H__
#define __UTILS_H__

inline int get_warp_size() {
  int gpu;
  cudaDeviceProp gpu_prop;

  // Leggo le proprietà del device per ottimizzare la bfs
  cudaGetDevice(&gpu);
  cudaGetDeviceProperties(&gpu_prop, gpu);

  return gpu_prop.warpSize;
}

// Utility per "cercare di" scegliere un numero ottimale di thread e blocchi
inline void set_threads_and_blocks(int *threads, int *blocks, int warp, UL queue_len, int thread_per_block) {
    int gpu, max_threads, max_blocks, num_blocks, num_threads, max_thread_per_block = 256;
    cudaDeviceProp gpu_prop;

    cudaGetDevice(&gpu);
    cudaGetDeviceProperties(&gpu_prop, gpu);

    max_threads = gpu_prop.maxThreadsPerBlock;
    max_blocks = gpu_prop.maxGridSize[0];

    num_threads = queue_len * warp;

    if(thread_per_block > 0 && thread_per_block <= max_threads) {
        num_blocks = num_threads / thread_per_block;
        if(num_threads % thread_per_block > 0) num_blocks++;
        if(num_blocks <= max_blocks) {
            *threads = thread_per_block;
            *blocks = num_blocks;
            return;
        }
        do {
            num_blocks /= 2;
        } while(num_blocks > max_blocks);
        *threads = thread_per_block;
        *blocks = num_blocks;
        return;
    }

    num_blocks = num_threads / max_thread_per_block;
    if(num_threads % max_thread_per_block > 0) num_blocks++;
    if(num_blocks <= max_blocks) {
        *threads = max_thread_per_block;
        *blocks = num_blocks;
        return;
    }
    do {
        num_blocks /= 2;
    } while(num_blocks > max_blocks);
    *threads = max_thread_per_block;
    *blocks = num_blocks;
    return;
}

#endif
