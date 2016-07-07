#ifndef __TIMERS_H__
#define __TIMERS_H__

// Funzione per far partire il timer
static inline void START_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop)
{
	HANDLE_ERROR(cudaEventCreate(start));
	HANDLE_ERROR(cudaEventCreate(stop));
	HANDLE_ERROR(cudaEventRecord(*start, 0));
}
// Funzione per fermare il timer e calcolare il tempo trascorso
static inline double STOP_CUDA_TIMER(cudaEvent_t *start, cudaEvent_t *stop)
{
    float time=0;
	HANDLE_ERROR(cudaEventRecord(*stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(*stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, *start, *stop));
	HANDLE_ERROR(cudaEventDestroy(*start));
	HANDLE_ERROR(cudaEventDestroy(*stop));
	return (time/1000.0);
}

#endif
