#pragma once
#include <cuda_runtime.h>

// Macro to check CUDA errors with file and line information
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

int cdiv(int x, int y) {
    return (x + y - 1) / y;
}
