#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <chrono>
#include "common.h"

// Scaled dot-product attention (single head):
//
//   Q : [M, d]   (queries)
//   K : [N, d]   (keys)
//   V : [N, d]   (values)
//   O : [M, d]   (output)
//
//   scores = (Q . K^T) / sqrt(d)        -> [M, N]
//   P      = softmax(scores, axis=-1)   -> [M, N]   (row-wise softmax)
//   O      = P . V                      -> [M, d]
//
// All matrices are row-major and stored as flat float arrays.

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------
void attentionCPU(const float* Q, const float* K, const float* V, float* O,
                  int M, int N, int d) {
    const float scale = 1.0f / sqrtf((float)d);
    float* scores = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < M; i++) {
        // scores[j] = scale * dot(Q[i], K[j]); track row max for numerical stability
        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += Q[i * d + k] * K[j * d + k];
            }
            dot *= scale;
            scores[j] = dot;
            if (dot > row_max) row_max = dot;
        }

        // softmax over the row
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            scores[j] = expf(scores[j] - row_max);
            sum += scores[j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; j++) {
            scores[j] *= inv_sum;
        }

        // O[i] = sum_j P[i][j] * V[j]
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++) {
                acc += scores[j] * V[j * d + k];
            }
            O[i * d + k] = acc;
        }
    }

    free(scores);
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Kernel 1: scores[i][j] = scale * dot(Q[i], K[j])
__global__ void attentionScoresKernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       float* __restrict__ scores,
                                       int M, int N, int d, float scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key index   (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index (row)
    if (i >= M || j >= N) return;

    float dot = 0.0f;
    const float* q = Q + i * d;
    const float* k = K + j * d;
    for (int e = 0; e < d; e++) {
        dot += q[e] * k[e];
    }
    scores[i * N + j] = dot * scale;
}

// Kernel 2: in-place row-wise softmax of scores ([M, N]). One block per row.
constexpr int SoftmaxRowThreads = 256;

__global__ void softmaxRowsKernel(float* __restrict__ scores, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    float* s = scores + row * N;
    int t = threadIdx.x;
    __shared__ float buf[SoftmaxRowThreads];

    // ---- row max ----
    float local_max = -FLT_MAX;
    for (int j = t; j < N; j += blockDim.x) local_max = fmaxf(local_max, s[j]);
    buf[t] = local_max;
    __syncthreads();
    for (int r = blockDim.x / 2; r >= 1; r >>= 1) {
        if (t < r) buf[t] = fmaxf(buf[t], buf[t + r]);
        __syncthreads();
    }
    float row_max = buf[0];
    __syncthreads();

    // ---- exp + row sum ----
    float local_sum = 0.0f;
    for (int j = t; j < N; j += blockDim.x) {
        float e = expf(s[j] - row_max);
        s[j] = e;
        local_sum += e;
    }
    buf[t] = local_sum;
    __syncthreads();
    for (int r = blockDim.x / 2; r >= 1; r >>= 1) {
        if (t < r) buf[t] += buf[t + r];
        __syncthreads();
    }
    float inv_sum = 1.0f / buf[0];
    __syncthreads();

    // ---- normalize ----
    for (int j = t; j < N; j += blockDim.x) s[j] *= inv_sum;
}

// Kernel 3: O[i][k] = sum_j P[i][j] * V[j][k]
__global__ void attentionOutputKernel(const float* __restrict__ P,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int M, int N, int d) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // head-dim index (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index    (row)
    if (i >= M || k >= d) return;

    float acc = 0.0f;
    const float* p = P + i * N;
    for (int j = 0; j < N; j++) {
        acc += p[j] * V[j * d + k];
    }
    O[i * d + k] = acc;
}

// ---------------------------------------------------------------------------
// Kernel launcher (with warmup / timing modes, mirroring the other kernels)
// ---------------------------------------------------------------------------
float launchAttentionKernel(const float* d_Q, const float* d_K, const float* d_V,
                            float* d_O, int M, int N, int d, bool warmup = false) {
    static bool config_printed = false;
    if (!warmup && !config_printed) {
        printf("Running GPU implementation...\n");
        config_printed = true;
    }

    const float scale = 1.0f / sqrtf((float)d);

    // Intermediate scores / probabilities buffer [M, N] (allocated once, reused).
    float* d_scores;
    CHECK_CUDA_ERROR(cudaMalloc(&d_scores, (size_t)M * N * sizeof(float)));

    dim3 scoreBlock(16, 16);
    dim3 scoreGrid(cdiv(N, scoreBlock.x), cdiv(M, scoreBlock.y));
    dim3 outBlock(32, 8);
    dim3 outGrid(cdiv(d, outBlock.x), cdiv(M, outBlock.y));

    auto run_kernels = [&]() {
        attentionScoresKernel<<<scoreGrid, scoreBlock>>>(d_Q, d_K, d_scores, M, N, d, scale);
        softmaxRowsKernel<<<M, SoftmaxRowThreads>>>(d_scores, M, N);
        attentionOutputKernel<<<outGrid, outBlock>>>(d_scores, d_V, d_O, M, N, d);
    };

    if (warmup) {
        const int warmup_iterations = 100;
        for (int i = 0; i < warmup_iterations; i++) run_kernels();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_scores));
        return 0.0f;
    }

    const int timing_trials = 1000;
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_stop));

    CHECK_CUDA_ERROR(cudaEventRecord(gpu_start));
    for (int trial = 0; trial < timing_trials; trial++) run_kernels();
    CHECK_CUDA_ERROR(cudaEventRecord(gpu_stop));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_stop));

    float total_gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_gpu_time_ms, gpu_start, gpu_stop));

    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_stop));
    CHECK_CUDA_ERROR(cudaFree(d_scores));

    return (total_gpu_time_ms * 1000.0f) / timing_trials;  // microseconds per trial
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
bool compareResults(const float* a, const float* b, int n, float tolerance = 1e-3f) {
    for (int i = 0; i < n; i++) {
        if (!isfinite(b[i]) || fabs(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f, diff = %f\n",
                   i, a[i], b[i], fabs(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

void initializeMatrix(float* m, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        m[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

void printSampleResults(const float* O, int M, int d, int samples = 5) {
    int cols = d < 6 ? d : 6;
    printf("\nSample output rows (first %d rows, first %d cols):\n", samples, cols);
    for (int i = 0; i < samples && i < M; i++) {
        printf("  row %d:", i);
        for (int k = 0; k < cols; k++) printf(" % .6f", O[i * d + k]);
        printf("%s\n", cols < d ? " ..." : "");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Defaults: M queries, N keys/values, d head dimension.
    int M = 512;
    int N = 512;
    int d = 64;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) d = atoi(argv[3]);

    if (M <= 0 || N <= 0 || d <= 0 || M > 8192 || N > 8192 || d > 1024) {
        printf("Invalid dimensions. Using defaults M=512, N=512, d=64.\n");
        M = 512; N = 512; d = 64;
    }

    printf("Scaled Dot-Product Attention - Q[%d,%d], K[%d,%d], V[%d,%d] -> O[%d,%d]\n",
           M, d, N, d, N, d, M, d);
    double qkv_mb = ((double)(M + 2 * N) * d * sizeof(float)) / (1024.0 * 1024.0);
    double scores_mb = ((double)M * N * sizeof(float)) / (1024.0 * 1024.0);
    printf("Q/K/V memory: %.2f MB, scores buffer: %.2f MB\n", qkv_mb, scores_mb);

    size_t q_size = (size_t)M * d * sizeof(float);
    size_t kv_size = (size_t)N * d * sizeof(float);
    size_t o_size = (size_t)M * d * sizeof(float);

    // Host allocations
    float* h_Q = (float*)malloc(q_size);
    float* h_K = (float*)malloc(kv_size);
    float* h_V = (float*)malloc(kv_size);
    float* h_O_cpu = (float*)malloc(o_size);
    float* h_O_gpu = (float*)malloc(o_size);
    if (!h_Q || !h_K || !h_V || !h_O_cpu || !h_O_gpu) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(1);
    }

    srand(time(NULL));
    printf("Initializing Q, K, V with random values in [-1, 1]...\n");
    initializeMatrix(h_Q, M, d);
    initializeMatrix(h_K, N, d);
    initializeMatrix(h_V, N, d);

    // ---- CPU reference ----
    // Scale the CPU trial count down for larger problems so the benchmark stays quick.
    long long cpu_work = (long long)M * N * d;
    int cpu_trials = (int)(200000000LL / cpu_work);
    if (cpu_trials < 5) cpu_trials = 5;
    if (cpu_trials > 200) cpu_trials = 200;
    int cpu_warmup_trials = cpu_trials < 10 ? cpu_trials : 10;
    printf("CPU warmup (%d trials)...\n", cpu_warmup_trials);
    for (int t = 0; t < cpu_warmup_trials; t++) attentionCPU(h_Q, h_K, h_V, h_O_cpu, M, N, d);

    printf("CPU timing runs (%d trials)...\n", cpu_trials);
    double total_cpu_time_us = 0.0;
    for (int t = 0; t < cpu_trials; t++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        attentionCPU(h_Q, h_K, h_V, h_O_cpu, M, N, d);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        total_cpu_time_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    }
    double cpu_time_us = total_cpu_time_us / cpu_trials;
    printf("CPU Time (averaged over %d trials): %.2f microseconds\n", cpu_trials, cpu_time_us);

    // ---- GPU ----
    printf("Allocating GPU memory...\n");
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Q, q_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K, kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V, kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O, o_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q, q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, h_K, kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, h_V, kv_size, cudaMemcpyHostToDevice));

    printf("GPU warmup (100 trials)...\n");
    launchAttentionKernel(d_Q, d_K, d_V, d_O, M, N, d, true);

    printf("GPU timing runs (1000 trials)...\n");
    float gpu_time_us = launchAttentionKernel(d_Q, d_K, d_V, d_O, M, N, d, false);
    printf("GPU Time (averaged over 1000 trials): %.2f microseconds\n", gpu_time_us);

    CHECK_CUDA_ERROR(cudaMemcpy(h_O_gpu, d_O, o_size, cudaMemcpyDeviceToHost));

    // ---- compare ----
    printf("\nComparing results...\n");
    if (compareResults(h_O_cpu, h_O_gpu, M * d)) {
        printf("✓ Results match! GPU implementation is correct.\n");
        if (gpu_time_us > 0) {
            printf("Speedup: %.2fx\n", cpu_time_us / gpu_time_us);
            // 2 matmuls (Q.K^T and P.V), each ~2*M*N*d flops
            double gflops = (4.0 * (double)M * N * d) / (gpu_time_us * 1e-6) / 1e9;
            printf("GPU throughput: %.2f GFLOP/s\n", gflops);
        }
        printSampleResults(h_O_cpu, M, d);
    } else {
        printf("✗ Results do not match! Check the GPU implementation.\n");
    }

    // ---- cleanup ----
    free(h_Q); free(h_K); free(h_V); free(h_O_cpu); free(h_O_gpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
