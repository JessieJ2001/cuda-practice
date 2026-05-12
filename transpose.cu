#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include "common.h"

// CPU reference implementation for matrix transpose
void transposeCPU(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

constexpr int TILE_SIZE = 32;

// TODO: Implement this GPU kernel for matrix transpose
// Hint: Consider memory coalescing patterns and shared memory tiling
__global__ void transposeNaive(const float* input, float* output, int rows, int cols) {
    // Simple approach - may have coalescing issues
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // blockDim == blockSize (32 * 32)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// TODO: Implement optimized version using shared memory tiling
__global__ void transposeShared(const float* input, float* output, int rows, int cols) {
    // Use shared memory to improve coalescing
    __shared__ float tile[(TILE_SIZE + 1) * TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
        tile[threadIdx.y * (TILE_SIZE + 1) + threadIdx.x] = input[row * cols + col];
    __syncthreads(); // all warps have to arrive before proceeding

    int outrow = blockIdx.x * blockDim.x + threadIdx.y;
    int outcol = blockIdx.y * blockDim.y + threadIdx.x;
    if (outrow < cols && outcol < rows) {
        float out = tile[threadIdx.x * (TILE_SIZE + 1) + threadIdx.y];

        output[outrow * rows + outcol] = out;
    }
}

// Function to launch the transpose kernel with appropriate configuration
float launchTransposeKernel(float* d_input, float* d_output, int rows, int cols, bool use_shared = true, bool warmup = false) {
    static bool config_printed = false;
    if (!warmup && !config_printed) {
        printf("Running GPU implementation (%s)...\n", use_shared ? "shared memory" : "naive");
        config_printed = true;
    }

    dim3 blockSize(TILE_SIZE, TILE_SIZE); // number of threads in a CTA
    // rows = 96, cols = 256
    // gridSize = (8, 3)
    dim3 gridSize(cdiv(cols, TILE_SIZE), cdiv(rows, TILE_SIZE)); // number of total CTAs
    
    auto run_kernel = [&]() {
        if (use_shared) {
            transposeShared<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
        } else {
            transposeNaive<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
        }
    };
    
    if (warmup) {
        // Warmup run - run a few iterations to warm up GPU
        const int warmup_iterations = 100;
        for (int i = 0; i < warmup_iterations; i++) {
            run_kernel();
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return 0.0f;
    }
    
    // Timing run - execute kernel loop 1000 times within single timing block
    const int timing_trials = 1000;
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(gpu_start));
    
    // Run kernel 1000 times in a loop
    for (int trial = 0; trial < timing_trials; trial++) {
        run_kernel();
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(gpu_stop));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_stop));
    
    float total_gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_gpu_time_ms, gpu_start, gpu_stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_stop));
    
    // Convert to microseconds and return average time per trial
    return (total_gpu_time_ms * 1000.0f) / timing_trials;
}

// Function to compare CPU and GPU results
bool compareResults(const float* cpu_result, const float* gpu_result, int rows, int cols, float tolerance = 1e-5f) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            int row = i / cols;
            int col = i % cols;
            printf("Mismatch at [%d,%d] (index %d): CPU = %f, GPU = %f, diff = %f\n", 
                   row, col, i, cpu_result[i], gpu_result[i], fabs(cpu_result[i] - gpu_result[i]));
            return false;
        }
    }
    return true;
}

// Function to initialize input matrix with test pattern
void initializeMatrix(float* matrix, int rows, int cols, bool use_pattern = true) {
    if (use_pattern) {
        // Use a predictable pattern for easy verification
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] = i * cols + j + 1.0f;  // 1-indexed pattern
            }
        }
    } else {
        // Random values
        srand(time(NULL));
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = (float)rand() / RAND_MAX * 100.0f;
        }
    }
}

// Function to print matrix (for small matrices)
void printMatrix(const float* matrix, int rows, int cols, const char* name, int max_display = 8) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    int display_rows = (rows < max_display) ? rows : max_display;
    int display_cols = (cols < max_display) ? cols : max_display;
    
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            printf("%6.1f ", matrix[i * cols + j]);
        }
        if (cols > max_display) printf("...");
        printf("\n");
    }
    if (rows > max_display) {
        printf("...\n");
    }
    printf("\n");
}

// Function to calculate memory bandwidth
void printBandwidthInfo(float time_us, int rows, int cols) {
    size_t bytes_read = rows * cols * sizeof(float);
    size_t bytes_written = rows * cols * sizeof(float);
    size_t total_bytes = bytes_read + bytes_written;
    
    double bandwidth_gbps = (total_bytes / (time_us * 1e-6)) / 1e9;
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth_gbps);
}

int main(int argc, char** argv) {
    // Parse command line arguments or use defaults
    int rows = 1024;  // Default matrix size
    int cols = 1024;
    bool use_pattern = false;
    bool use_shared = true;
    
    int device;
    cudaGetDevice(&device);  // get current device

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    
    if (argc > 1) {
        rows = atoi(argv[1]);
        if (rows <= 0 || rows > 8192) {
            printf("Invalid number of rows. Using default size of 1024.\n");
            rows = 1024;
        }
    }
    
    if (argc > 2) {
        cols = atoi(argv[2]);
        if (cols <= 0 || cols > 8192) {
            printf("Invalid number of columns. Using default size of 1024.\n");
            cols = 1024;
        }
    }
    
    if (argc > 3 && strcmp(argv[3], "pattern") == 0) {
        use_pattern = true;
        printf("Using test pattern for verification\n");
    }
    
    if (argc > 4 && strcmp(argv[4], "naive") == 0) {
        use_shared = false;
        printf("Using naive implementation (no shared memory)\n");
    }
    
    printf("Matrix Transpose - %dx%d matrix\n", rows, cols);
    printf("Matrix size: %.2f MB\n", (rows * cols * sizeof(float)) / (1024.0f * 1024.0f));
    
    // Calculate memory sizes
    size_t input_size = rows * cols * sizeof(float);
    size_t output_size = cols * rows * sizeof(float);  // Transposed dimensions
    
    // Allocate host memory
    float* h_input = (float*)malloc(input_size);
    float* h_output_cpu = (float*)malloc(output_size);
    float* h_output_gpu = (float*)malloc(output_size);
    
    if (!h_input || !h_output_cpu || !h_output_gpu) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(1);
    }
    
    // Initialize input matrix
    printf("Initializing input matrix...\n");
    initializeMatrix(h_input, rows, cols, use_pattern);
    
    // Print input matrix if small enough
    if (rows <= 8 && cols <= 8) {
        printMatrix(h_input, rows, cols, "Input Matrix");
    }
    
    // CPU warmup (100 trials)
    printf("CPU warmup (100 trials)...\n");
    const int cpu_warmup_trials = 100;
    for (int trial = 0; trial < cpu_warmup_trials; trial++) {
        transposeCPU(h_input, h_output_cpu, rows, cols);
    }
    
    // Run CPU reference implementation (100 trials)
    printf("CPU timing runs (100 trials)...\n");
    double total_cpu_time_us = 0.0;
    const int cpu_trials = 100;
    
    for (int trial = 0; trial < cpu_trials; trial++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        transposeCPU(h_input, h_output_cpu, rows, cols);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        total_cpu_time_us += duration.count();
    }
    double cpu_time_microseconds = total_cpu_time_us / cpu_trials;
    printf("CPU Time (averaged over %d trials): %.2f microseconds\n", cpu_trials, cpu_time_microseconds);
    printBandwidthInfo(cpu_time_microseconds, rows, cols);
    
    // Print CPU result if small enough
    if (rows <= 8 && cols <= 8) {
        printMatrix(h_output_cpu, cols, rows, "CPU Result");
    }
    
    // Allocate device memory
    printf("Allocating GPU memory...\n");
    float* d_input, * d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    // GPU warmup (100 trials)
    printf("GPU warmup (100 trials)...\n");
    launchTransposeKernel(d_input, d_output, rows, cols, use_shared, true);  // warmup = true
    
    // GPU timing runs (1000 trials)
    printf("GPU timing runs (1000 trials)...\n");
    float gpu_time_microseconds = launchTransposeKernel(d_input, d_output, rows, cols, use_shared, false);  // warmup = false
    printf("GPU Time (averaged over 1000 trials): %.2f microseconds\n", gpu_time_microseconds);
    printBandwidthInfo(gpu_time_microseconds, rows, cols);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Print GPU result if small enough
    if (rows <= 8 && cols <= 8) {
        printMatrix(h_output_gpu, cols, rows, "GPU Result");
    }
    
    // Compare results
    printf("\nComparing results...\n");
    if (compareResults(h_output_cpu, h_output_gpu, cols, rows)) {  // Note: transposed dimensions
        printf("✓ Results match! GPU implementation is correct.\n");
        
        // Print performance comparison
        if (gpu_time_microseconds > 0) {
            double speedup = cpu_time_microseconds / gpu_time_microseconds;
            printf("Speedup: %.2fx\n", speedup);
        }
    } else {
        printf("✗ Results do not match! Check your GPU implementation.\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
} 