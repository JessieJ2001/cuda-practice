#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <chrono>
#include "common.h"

// CPU reference implementation for softmax
void softmaxCPU(const float* input, float* output, int N) {
    // Step 1: Find maximum value (max trick to prevent overflow)
    float max_val = input[0];
    for (int i = 1; i < N; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Step 2: Compute exp(x_i - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Step 3: Normalize by sum
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

constexpr int SoftmaxThreadsPerBlock{256};
constexpr int SoftmaxValuesPerBlock = SoftmaxThreadsPerBlock * 2;

// TODO: Implement this GPU kernel for softmax
// Hint: You'll need multiple steps - consider using multiple kernel launches
// or shared memory for reductions
__global__ void softmaxPartial(const float* input, const float* max_val, float* exp_sum, int N) {
    // 1. Finding the maximum value (reduction)
    // 2. Computing exp(x_i - max) for each element
    // 3. Computing the sum of all exponentials (reduction)
    // 4. Normalizing each element by the sum
    __shared__ float data[SoftmaxThreadsPerBlock];

    float unnorm = 0;
    float maxv = max_val[0];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        unnorm += exp(input[idx] - maxv);
    
    int next_idx = idx + blockDim.x;
    if (next_idx < N)   
        unnorm += exp(input[next_idx] - maxv);
    
    data[threadIdx.x] = unnorm;
    // perform sum reduction in-block
    for (int range = blockDim.x / 2; range >= 1; range /= 2) {
        if (threadIdx.x < range)
            data[threadIdx.x] = data[threadIdx.x] + data[threadIdx.x + range];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float value = data[0];
        atomicAdd(exp_sum, value);
    }
}

__global__ void softmaxNormalize(const float* input, const float* exp_sum, const float* max_val, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float value = exp(input[idx] - max_val[0]);

    output[idx] = value / exp_sum[0];
}

constexpr int ReduceThreadsPerBlock{256};
constexpr int ReduceValuesPerBlock = ReduceThreadsPerBlock * 2;

// Helper function for finding maximum on GPU (you may want to implement this)
template <typename Func>
__forceinline__ __device__ void reduceKernel(const float* input, float* result, int N, Func&& red_func) {
    // TODO: Implement parallel reduction to find maximum
    __shared__ float data[ReduceThreadsPerBlock];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;
    if (idx < N)
        value = input[idx];
    int next_idx = idx + blockDim.x;
    if (next_idx < N)
        value = input[next_idx];

    data[threadIdx.x] = value;

    // OPTIMIZE warp reduce
    #pragma unroll
    for (int range = blockDim.x / 2; range >= 1; range /= 2) {
        if (threadIdx.x < range)
            data[threadIdx.x] = red_func(data[threadIdx.x], data[threadIdx.x + range]);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float value = data[0];
        atomicAdd(result, value);
    }
}

__global__ void findMaxKernel(const float* input, float* result, int N) {
    reduceKernel(input, result, N, [](auto&& a, auto&& b) { return max(a, b); });
}

// Function to launch the softmax kernel with appropriate configuration
float launchSoftmaxKernel(float* d_input, float* d_output, int N, bool warmup = false) {
    static bool config_printed = false;
    if (!warmup && !config_printed) {
        printf("Running GPU implementation...\n");
        config_printed = true;
    }

    // Allocate memory once outside the timing loop
    float* d_max, *d_sum;
    CHECK_CUDA_ERROR(cudaMalloc(&d_max, 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum, 1 * sizeof(float)));
    
    auto run_kernels = [&]() {
        // Reset values for each run
        CHECK_CUDA_ERROR(cudaMemset(d_max, 0, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(float)));
        
        findMaxKernel<<<cdiv(N, ReduceValuesPerBlock), ReduceThreadsPerBlock>>>(d_input, d_max, N);
        softmaxPartial<<<cdiv(N, SoftmaxValuesPerBlock), SoftmaxThreadsPerBlock>>>(d_input, d_max, d_sum, N);
        int normalizeThreads = 256;
        softmaxNormalize<<<cdiv(N, normalizeThreads), normalizeThreads>>>(d_input, d_sum, d_max, d_output, N);
    };
    
    if (warmup) {
        // Warmup run - run a few iterations to warm up GPU
        const int warmup_iterations = 100;
        for (int i = 0; i < warmup_iterations; i++) {
            run_kernels();
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_max));
        CHECK_CUDA_ERROR(cudaFree(d_sum));
        return 0.0f;
    }
    
    // Timing run - execute kernel loop 1000 times within single timing block
    const int timing_trials = 1000;
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(gpu_start));
    
    // Run kernels 1000 times in a loop
    for (int trial = 0; trial < timing_trials; trial++) {
        run_kernels();
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(gpu_stop));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_stop));
    
    float total_gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_gpu_time_ms, gpu_start, gpu_stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_stop));
    CHECK_CUDA_ERROR(cudaFree(d_max));
    CHECK_CUDA_ERROR(cudaFree(d_sum));
    
    // Convert to microseconds and return average time per trial
    return (total_gpu_time_ms * 1000.0f) / timing_trials;
}

// Function to compare CPU and GPU results
bool compareResults(const float* cpu_result, const float* gpu_result, int N, float tolerance = 1e-4f) {
    for (int i = 0; i < N; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f, diff = %f\n", 
                   i, cpu_result[i], gpu_result[i], fabs(cpu_result[i] - gpu_result[i]));
            return false;
        }
    }
    return true;
}

// Function to verify softmax properties
bool verifySoftmax(const float* output, int N, float tolerance = 1e-4f) {
    // Check that all values are positive
    for (int i = 0; i < N; i++) {
        if (output[i] < 0.0f) {
            printf("Negative value at index %d: %f\n", i, output[i]);
            return false;
        }
    }
    
    // Check that sum is approximately 1.0
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += output[i];
    }
    
    if (fabs(sum - 1.0f) > tolerance) {
        printf("Sum is not 1.0: %f (difference: %f)\n", sum, fabs(sum - 1.0f));
        return false;
    }
    
    return true;
}

// Function to initialize input with random or test values
void initializeInput(float* input, int N, bool use_test_case = false) {
    if (use_test_case && N >= 3) {
        // Use the example from the problem
        input[0] = 1.0f;
        input[1] = 2.0f;
        input[2] = 3.0f;
        for (int i = 3; i < N; i++) {
            input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;  // Random values [-5, 5]
        }
    } else {
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            input[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;  // Random values [-10, 10]
        }
    }
}

// Function to print sample results
void printSampleResults(const float* input, const float* output, int N, int samples = 10) {
    printf("\nSample results (first %d elements):\n", samples);
    printf("Index\tInput\t\tSoftmax\n");
    printf("-----\t-----\t\t-------\n");
    for (int i = 0; i < samples && i < N; i++) {
        printf("%d\t%.3f\t\t%.6f\n", i, input[i], output[i]);
    }
    
    // Verify sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += output[i];
    }
    printf("\nSum of all softmax values: %.6f (should be ~1.0)\n", sum);
}

int main(int argc, char** argv) {
    // Parse command line arguments or use default
    int N = 10000;  // Default array size
    bool use_test_case = false;
    
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N > 500000) {
            printf("Invalid array size. Using default size of 10,000.\n");
            N = 10000;
        }
    }
    
    if (argc > 2 && strcmp(argv[2], "test") == 0) {
        use_test_case = true;
        printf("Using test case values for first 3 elements\n");
    }
    
    printf("Softmax Computation - N = %d elements\n", N);
    printf("Array size: %.2f MB\n", (N * sizeof(float)) / (1024.0f * 1024.0f));
    
    // Calculate memory sizes
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output_cpu = (float*)malloc(size);
    float* h_output_gpu = (float*)malloc(size);
    
    if (!h_input || !h_output_cpu || !h_output_gpu) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(1);
    }
    
    // Initialize input array
    printf("Initializing input array...\n");
    initializeInput(h_input, N, use_test_case);
    
    // CPU warmup (1000 trials)
    printf("CPU warmup (1000 trials)...\n");
    const int cpu_warmup_trials = 1000;
    for (int trial = 0; trial < cpu_warmup_trials; trial++) {
        softmaxCPU(h_input, h_output_cpu, N);
    }
    
    // Run CPU reference implementation (1000 trials)
    printf("CPU timing runs (1000 trials)...\n");
    double total_cpu_time_us = 0.0;
    const int cpu_trials = 1000;
    
    for (int trial = 0; trial < cpu_trials; trial++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        softmaxCPU(h_input, h_output_cpu, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        total_cpu_time_us += duration.count();
    }
    double cpu_time_microseconds = total_cpu_time_us / cpu_trials;
    printf("CPU Time (averaged over %d trials): %.2f microseconds\n", cpu_trials, cpu_time_microseconds);
    
    // Verify CPU result
    if (!verifySoftmax(h_output_cpu, N)) {
        printf("CPU implementation failed verification!\n");
        exit(1);
    }
    printf("✓ CPU result verified (sum = 1, all positive)\n");
    
    // Allocate device memory
    printf("Allocating GPU memory...\n");
    float* d_input, * d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // GPU warmup (100 trials)
    printf("GPU warmup (100 trials)...\n");
    launchSoftmaxKernel(d_input, d_output, N, true);  // warmup = true (handles 100 trials internally)
    
    // GPU timing runs (1000 trials)
    printf("GPU timing runs (1000 trials)...\n");
    float gpu_time_microseconds = launchSoftmaxKernel(d_input, d_output, N, false);  // warmup = false (handles 1000 trials internally)
    printf("GPU Time (averaged over 1000 trials): %.2f microseconds\n", gpu_time_microseconds);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    // Compare results
    printf("\nComparing results...\n");
    
    // First verify GPU result properties
    if (!verifySoftmax(h_output_gpu, N)) {
        printf("✗ GPU result failed verification (sum != 1 or negative values)!\n");
    } else {
        printf("✓ GPU result verified (sum = 1, all positive)\n");
        
        // Compare with CPU result
        if (compareResults(h_output_cpu, h_output_gpu, N)) {
            printf("✓ Results match! GPU implementation is correct.\n");
            
            // Print performance comparison
            if (gpu_time_microseconds > 0) {
                double speedup = cpu_time_microseconds / gpu_time_microseconds;  // Both in microseconds
                printf("Speedup: %.2fx\n", speedup);
            }
            
            // Print sample results
            printSampleResults(h_input, h_output_cpu, N);
        } else {
            printf("✗ Results do not match! Check your GPU implementation.\n");
        }
    }
    
    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
} 