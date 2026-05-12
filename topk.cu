#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include "common.h"

// Structure to hold value and its original index
struct ValueIndex {
    float value;
    int index;
    
    __host__ __device__ bool operator<(const ValueIndex& other) const {
        return value > other.value;  // For max-heap behavior (largest first)
    }
    
    __host__ __device__ bool operator>(const ValueIndex& other) const {
        return value < other.value;  // For min-heap behavior
    }
};

// CPU reference implementation using std::sort
void topkCPU(const float* input, ValueIndex* output, int N, int K) {
    // Create value-index pairs
    std::vector<ValueIndex> pairs(N);
    for (int i = 0; i < N; i++) {
        pairs[i] = {input[i], i};
    }
    
    // Sort in descending order (largest first)
    std::sort(pairs.begin(), pairs.end());
    
    // Copy top K elements
    for (int i = 0; i < K && i < N; i++) {
        output[i] = pairs[i];
    }
}

// CPU reference using partial sort (more efficient for small K)
void topkPartialSortCPU(const float* input, ValueIndex* output, int N, int K) {
    // Create value-index pairs
    std::vector<ValueIndex> pairs(N);
    for (int i = 0; i < N; i++) {
        pairs[i] = {input[i], i};
    }
    
    // Partial sort - only sort the first K elements
    std::partial_sort(pairs.begin(), pairs.begin() + K, pairs.end());
    
    // Copy top K elements
    for (int i = 0; i < K && i < N; i++) {
        output[i] = pairs[i];
    }
}

constexpr int RadixBits = 4;
constexpr uint32_t RadixMask = (1 << (RadixBits + 1)) - 1;
constexpr int NumBuckets = RadixMask + 1;
constexpr int RadixBlockSize = 256;


__global__ void zero(int* hist) {
    if (threadIdx.x < NumBuckets)
        hist[threadIdx.x] = 0;
}


// hist = (num_buckets x num_blocks) for coalesced loading
// Hint: Use radix sort to find the K-th largest element, then collect all elements >= threshold
__global__ void radixSelectk(const float* input, int* output, int N, int prefix_val, int radix_offset) {
    __shared__ int hist[NumBuckets];

    if (threadIdx.x < NumBuckets) {
        hist[threadIdx.x] = 0;
    }
    __syncthreads();

    int shift = 32 - RadixBits - radix_offset;
    // if radix_offset == 4, this is 0b1111 << 28
    // if radix_offset == 8, this is 0b11111111 << 24
    uint32_t prefix_mask = ((1 << radix_offset) - 1) << (32 - radix_offset);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float val = input[i];
        uint32_t uval = reinterpret_cast<uint32_t&>(val);

        // not in bucket
        if ((uval & prefix_mask) != prefix_val) {
            continue;
        }

        auto bucket = (uval >> shift) & RadixMask;
        // NOTE: spin wait and bank conflict
        atomicAdd(&hist[bucket], 1);
    }
    __syncthreads();

    // uncoalesced but done only once
    for (int histid = threadIdx.x; histid < NumBuckets; histid += blockDim.x) {
        atomicAdd(&output[histid], hist[histid]);
    }
}

__global__ void locateValue(const float* input, float* selectk, int N, int prefix_val, int radix_offset) {
    uint32_t prefix_mask = ((1 << radix_offset) - 1) << (32 - radix_offset);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float val = input[i];
        uint32_t uval = reinterpret_cast<uint32_t&>(val);

        // not in bucket
        if ((uval & prefix_mask) == prefix_val) {
            selectk[0] = val;
            break;
        }
    }
}

constexpr int FilterBlockSize = 256;

__global__ void filterK(const float *input, ValueIndex *output, int *p_out_idx, int N, float selectk) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float val = input[i];

        if (val >= selectk) {
            auto out_idx = atomicAdd(p_out_idx, 1);
            output[out_idx] = ValueIndex{val, i};
        }
    }
}

// Function to launch the topk kernel with appropriate configuration
float launchTopKKernel(float* d_input, ValueIndex* d_output, int N, int K, bool warmup = false) {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int num_sms = prop.multiProcessorCount;


    static bool config_printed = false;
    if (!warmup && !config_printed) {
        printf("Running GPU implementation (radix select approach)...\n");
        config_printed = true;
    }

    int numBlocks = num_sms * 4;
    int *d_hist;
    float *d_selectk;
    CHECK_CUDA_ERROR(cudaMalloc(&d_hist, NumBuckets * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_selectk, sizeof(float)));

    int numFilterBlocks = num_sms * 4;
    int *d_topk_idx;
    CHECK_CUDA_ERROR(cudaMalloc(&d_topk_idx, 1));

    auto run_kernel = [&]() {
        // start from smallest
        int cur_k = N + 1 - K;
        int prefix_val = 0;
        int h_hist[NumBuckets];

        bool located = false;
        for (int radix_offset = 0; radix_offset < 32; radix_offset += RadixBits) {
            zero<<<1, 256>>>(d_hist);
            // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            // CHECK_CUDA_ERROR(cudaGetLastError());
            // zero d_hist_interm
            radixSelectk<<<numBlocks, RadixBlockSize>>>(d_input, d_hist, N, prefix_val, radix_offset);
            // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            // CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaMemcpy(h_hist, d_hist, NumBuckets * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            bool bucket_found = false;
            int cur = 0;
            for (int b = 0; b < NumBuckets; b++) {
                int next_cur = cur + h_hist[b];
                if (next_cur >= cur_k) {
                    bucket_found = true;
                    // found it
                    prefix_val |= b << (32 - RadixBits - radix_offset);
                    cur_k -= cur;

                    // found when bucket has only one element or the bucket contains identical values,
                    // i.e. radix_offset + RadixBits == 32
                    if (h_hist[b] == 1 || radix_offset + RadixBits == 32) {
                        located = true;
                        locateValue<<<numBlocks, RadixBlockSize>>>(d_input, d_selectk,
                            N, prefix_val, radix_offset + RadixBits);
                    }
                    break;
                }
                cur = next_cur;
            }

            if (!bucket_found) {
                exit(1);
            }

            if (located) {
                break;
            }
        }

        // can fuse copy to host in kernel
        float h_selectk;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_selectk, d_selectk, sizeof(float), cudaMemcpyDeviceToHost));
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // top-k
        zero<<<1, 32>>>(d_topk_idx);
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // CHECK_CUDA_ERROR(cudaGetLastError());
        filterK<<<numFilterBlocks, FilterBlockSize>>>(d_input, d_output, d_topk_idx, N, h_selectk);
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // CHECK_CUDA_ERROR(cudaGetLastError());
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

    cudaFree(d_hist);
    cudaFree(d_selectk);
    
    // Convert to microseconds and return average time per trial
    return (total_gpu_time_ms * 1000.0f) / timing_trials;
}

// Function to compare CPU and GPU results (order within top-K may differ)
bool compareTopKResults(const ValueIndex* cpu_result, const ValueIndex* gpu_result, int K, float tolerance = 1e-5f) {
    // Create sets of values for comparison (since order within top-K may vary)
    std::vector<float> cpu_values, gpu_values;
    std::vector<int> cpu_indices, gpu_indices;
    
    for (int i = 0; i < K; i++) {
        cpu_values.push_back(cpu_result[i].value);
        cpu_indices.push_back(cpu_result[i].index);
        gpu_values.push_back(gpu_result[i].value);
        gpu_indices.push_back(gpu_result[i].index);
    }
    
    // Sort both for comparison
    std::sort(cpu_values.begin(), cpu_values.end(), std::greater<float>());
    std::sort(gpu_values.begin(), gpu_values.end(), std::greater<float>());
    
    // Compare values
    for (int i = 0; i < K; i++) {
        if (fabs(cpu_values[i] - gpu_values[i]) > tolerance) {
            printf("Value mismatch at position %d: CPU = %f, GPU = %f\n", 
                   i, cpu_values[i], gpu_values[i]);
            return false;
        }
    }
    
    // Check that all GPU indices are valid and correspond to correct values
    for (int i = 0; i < K; i++) {
        if (gpu_result[i].index < 0 || gpu_result[i].index >= 10000000) {  // Reasonable bounds check
            printf("Invalid index at position %d: %d\n", i, gpu_result[i].index);
            return false;
        }
    }
    
    return true;
}

// Function to initialize input with test values
void initializeInput(float* input, int N, bool use_pattern = false) {
    // Create a pattern where we know the top K values
    srand(42);  // Fixed seed for reproducibility
    if (use_pattern) {
        for (int i = 0; i < N; i++) {
            input[i] = (float)(rand() % 1000) / 10.0f;  // Random values [0, 100)
        }
        
        // Ensure some known large values for verification
        if (N >= 10) {
            input[0] = 999.0f;   // Largest
            input[1] = 998.0f;   // Second largest
            input[2] = 997.0f;   // Third largest
            input[N-1] = 996.0f; // Fourth largest
            input[N/2] = 995.0f; // Fifth largest
        }
    } else {
        for (int i = 0; i < N; i++) {
            input[i] = (float)rand() / RAND_MAX * 1000.0f;
        }
    }
}

// Function to print top K results
void printTopKResults(const ValueIndex* result, int K, const char* label) {
    printf("\n%s Top-%d results:\n", label, K);
    printf("Rank\tValue\t\tIndex\n");
    printf("----\t-----\t\t-----\n");
    for (int i = 0; i < K; i++) {
        printf("%d\t%.2f\t\t%d\n", i+1, result[i].value, result[i].index);
    }
}

// Function to verify that results are actually top K
bool verifyTopK(const float* input, const ValueIndex* result, int N, int K) {
    // Find the K-th largest value in the result
    float kth_value = result[K-1].value;
    
    // Count how many elements in the original array are larger than kth_value
    int count_larger = 0;
    for (int i = 0; i < N; i++) {
        if (input[i] > kth_value) {
            count_larger++;
        }
    }
    
    // There should be at most K-1 elements larger than kth_value
    if (count_larger >= K) {
        printf("Verification failed: found %d elements larger than %d-th value (%.2f)\n", 
               count_larger, K, kth_value);
        return false;
    }
    
    // Verify that all result values are from the input array
    for (int i = 0; i < K; i++) {
        int idx = result[i].index;
        if (idx < 0 || idx >= N) {
            printf("Invalid index %d at position %d\n", idx, i);
            return false;
        }
        if (fabs(input[idx] - result[i].value) > 1e-5f) {
            printf("Value mismatch: input[%d] = %f but result claims %f\n", 
                   idx, input[idx], result[i].value);
            return false;
        }
    }
    
    return true;
}

int main(int argc, char** argv) {
    // Parse command line arguments or use defaults
    int N = 10000;    // Default array size
    int K = 10;       // Default K value
    bool use_pattern = false;
    
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N > 10000000) {
            printf("Invalid array size. Using default size of 10,000.\n");
            N = 10000;
        }
    }
    
    if (argc > 2) {
        K = atoi(argv[2]);
        if (K <= 0 || K > 1000000) {
            printf("Invalid K value. Using default K of 10.\n");
            K = 10;
        }
    }
    
    if (argc > 3 && strcmp(argv[3], "pattern") == 0) {
        use_pattern = true;
        printf("Using test pattern with known top values\n");
    }
    
    printf("Top-K Selection - N = %d elements, K = %d\n", N, K);
    printf("Array size: %.2f MB\n", (N * sizeof(float)) / (1024.0f * 1024.0f));
    printf("K/N ratio: %.4f\n", (float)K / N);
    
    // Calculate memory sizes
    size_t input_size = N * sizeof(float);
    size_t output_size = K * sizeof(ValueIndex);
    
    // Allocate host memory
    float* h_input = (float*)malloc(input_size);
    ValueIndex* h_output_cpu = (ValueIndex*)malloc(output_size);
    ValueIndex* h_output_gpu = (ValueIndex*)malloc(output_size);
    
    if (!h_input || !h_output_cpu || !h_output_gpu) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(1);
    }
    
    // Initialize input array
    printf("Initializing input array...\n");
    initializeInput(h_input, N, use_pattern);
    
    // Print sample input if small enough
    if (N <= 20) {
        printf("Input: ");
        for (int i = 0; i < N; i++) {
            printf("%.1f ", h_input[i]);
        }
        printf("\n");
    }
    
    // CPU warmup (100 trials)
    printf("CPU warmup (100 trials)...\n");
    const int cpu_warmup_trials = 100;
    for (int trial = 0; trial < cpu_warmup_trials; trial++) {
        topkPartialSortCPU(h_input, h_output_cpu, N, K);
    }
    
    // Run CPU reference implementation (100 trials)
    printf("CPU timing runs (100 trials)...\n");
    double total_cpu_time_us = 0.0;
    const int cpu_trials = 100;
    
    for (int trial = 0; trial < cpu_trials; trial++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        topkPartialSortCPU(h_input, h_output_cpu, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        total_cpu_time_us += duration.count();
    }
    double cpu_time_microseconds = total_cpu_time_us / cpu_trials;
    printf("CPU Time (averaged over %d trials): %.2f microseconds\n", cpu_trials, cpu_time_microseconds);
    
    // Verify CPU result
    if (!verifyTopK(h_input, h_output_cpu, N, K)) {
        printf("CPU implementation failed verification!\n");
        exit(1);
    }
    printf("✓ CPU result verified\n");
    
    // Print CPU result
    printTopKResults(h_output_cpu, K, "CPU");
    
    // Allocate device memory
    printf("Allocating GPU memory...\n");
    float* d_input;
    ValueIndex* d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    // GPU warmup (100 trials)
    printf("GPU warmup (100 trials)...\n");
    launchTopKKernel(d_input, d_output, N, K, true);  // warmup = true
    
    // GPU timing runs (1000 trials)
    printf("GPU timing runs (1000 trials)...\n");
    float gpu_time_microseconds = launchTopKKernel(d_input, d_output, N, K, false);  // warmup = false
    printf("GPU Time (averaged over 1000 trials): %.2f microseconds\n", gpu_time_microseconds);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Print GPU result
    printTopKResults(h_output_gpu, K, "GPU");
    
    // Compare results
    printf("\nComparing results...\n");
    
    // First verify GPU result properties
    if (!verifyTopK(h_input, h_output_gpu, N, K)) {
        printf("✗ GPU result failed verification!\n");
    } else {
        printf("✓ GPU result verified\n");
        
        // Compare with CPU result (allowing for different orderings within top-K)
        if (compareTopKResults(h_output_cpu, h_output_gpu, K)) {
            printf("✓ Results match! GPU implementation is correct.\n");
            
            // Print performance comparison
            if (gpu_time_microseconds > 0) {
                double speedup = cpu_time_microseconds / gpu_time_microseconds;
                printf("Speedup: %.2fx\n", speedup);
            }
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