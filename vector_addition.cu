#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include "common.h"

// CPU reference implementation for vector addition
void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// TODO: Implement this GPU kernel
// Hint: Use threadIdx.x, blockIdx.x, and blockDim.x to calculate the global thread index
__global__ void vectorAddGPU(const float* A, const float* B, float* C, int N) {
    // TODO: Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Function to launch the vector addition kernel with appropriate configuration
float launchVectorAddKernel(float* d_A, float* d_B, float* d_C, int N, bool warmup = false) {
    static bool config_printed = false;
    if (!warmup && !config_printed) {
        printf("Running GPU implementation...\n");
        int threadsPerBlock = 1024;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        printf("Grid size: %d blocks, Block size: %d threads\n", blocksPerGrid, threadsPerBlock);
        config_printed = true;
    }
    
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    if (warmup) {
        // Warmup run - no timing
        vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return 0.0f;
    }
    
    // Timing run
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(gpu_start));
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(gpu_stop));
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_stop));
    
    float gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_stop));
    
    return gpu_time_ms * 1000.0f;  // Convert to microseconds
}

// Function to compare CPU and GPU results
bool compareResults(const float* cpu_result, const float* gpu_result, int N, float tolerance = 1e-5f) {
    for (int i = 0; i < N; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

// Function to initialize vectors with random values
void initializeVectors(float* A, float* B, int N) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        A[i] = (float)rand() / RAND_MAX * 10.0f;  // Random values between 0 and 10
        B[i] = (float)rand() / RAND_MAX * 10.0f;
    }
}

// Function to print a small portion of vectors for verification
void printSampleResults(const float* A, const float* B, const float* C, int N, int samples = 5) {
    printf("\nSample results (first %d elements):\n", samples);
    printf("Index\tA\t\tB\t\tC\n");
    printf("-----\t---\t\t---\t\t---\n");
    for (int i = 0; i < samples && i < N; i++) {
        printf("%d\t%.3f\t\t%.3f\t\t%.3f\n", i, A[i], B[i], C[i]);
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments or use default
    int N = 1_000_000;  // Default vector size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N > 1_000_000_000) {
            printf("Invalid vector size. Using default size of 1,000,000.\n");
            N = 1_000_000;
        }
    }
    
    printf("Vector Addition - N = %d elements\n", N);
    printf("Vector size: %.2f MB\n", (N * sizeof(float)) / (1024.0f * 1024.0f));
    
    // Calculate memory sizes
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);
    
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(1);
    }
    
    // Initialize input vectors
    printf("Initializing vectors...\n");
    initializeVectors(h_A, h_B, N);
    
    // CPU warmup (1000 trials)
    printf("CPU warmup (1000 trials)...\n");
    const int cpu_warmup_trials = 1000;
    for (int trial = 0; trial < cpu_warmup_trials; trial++) {
        vectorAddCPU(h_A, h_B, h_C_cpu, N);
    }
    
    // Run CPU reference implementation (1000 trials)
    printf("CPU timing runs (1000 trials)...\n");
    double total_cpu_time_us = 0.0;
    const int cpu_trials = 1000;
    
    for (int trial = 0; trial < cpu_trials; trial++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        vectorAddCPU(h_A, h_B, h_C_cpu, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        total_cpu_time_us += duration.count();
    }
    double cpu_time_microseconds = total_cpu_time_us / cpu_trials;
    printf("CPU Time (averaged over %d trials): %.2f microseconds\n", cpu_trials, cpu_time_microseconds);
    
    // Allocate device memory
    printf("Allocating GPU memory...\n");
    float* d_A, * d_B, * d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));
    
    // Copy input vectors to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // GPU warmup (1000 trials)
    printf("GPU warmup (1000 trials)...\n");
    const int warmup_trials = 1000;
    for (int trial = 0; trial < warmup_trials; trial++) {
        launchVectorAddKernel(d_A, d_B, d_C, N, true);  // warmup = true
    }
    
    // GPU timing runs (1000 trials)
    printf("GPU timing runs (1000 trials)...\n");
    const int gpu_trials = 1000;
    float total_gpu_time_us = 0.0f;
    
    for (int trial = 0; trial < gpu_trials; trial++) {
        total_gpu_time_us += launchVectorAddKernel(d_A, d_B, d_C, N, false);  // warmup = false
    }
    
    float gpu_time_microseconds = total_gpu_time_us / gpu_trials;
    printf("GPU Time (averaged over %d trials): %.2f microseconds\n", gpu_trials, gpu_time_microseconds);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    
    // Compare results
    printf("\nComparing results...\n");
    if (compareResults(h_C_cpu, h_C_gpu, N)) {
        printf("✓ Results match! GPU implementation is correct.\n");
        
        // Print performance comparison
        if (gpu_time_microseconds > 0) {
            double speedup = cpu_time_microseconds / gpu_time_microseconds;  // Both in microseconds
            printf("Speedup: %.2fx\n", speedup);
        }
        
        // Print sample results
        printSampleResults(h_A, h_B, h_C_cpu, N);
    } else {
        printf("✗ Results do not match! Check your GPU implementation.\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}