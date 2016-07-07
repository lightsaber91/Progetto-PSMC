#ifndef __ERROR_H__
#define __ERROR_H__

// Funzione di controllo per vedere se si va in errore sul device
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static inline void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}

#endif
