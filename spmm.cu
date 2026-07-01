#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include "common.h"

// =====================================================================
// Sparse Matrix (A, M×N, ~60-70% zeros) × Dense Matrix (B, N×K)  ->  C (M×K)
// All matrices row-major, 32-bit float. GPU-native only (no cuSPARSE).
//
// Three strategies, benchmarked side by side:
//   1. spmmDenseNaive  — full dense GEMM, sparsity ignored (baseline)
//   2. spmmZeroSkip     — skip the FMA + B-row read when A==0  (solve uses this)
//   3. CSR build + spmmCSR — convert dense A to CSR, then multiply
//
// The required entry point `solve` (LeetGPU signature, device pointers,
// result in C) uses strategy 2: it is the fastest option that needs NO
// format conversion and NO extra memory, and — see docs/spmm.html — its
// zero-test is warp-uniform, so it is completely divergence-free.
// =====================================================================

#define TILE 16   // 16×16 = 256 threads per block (x = K columns, y = M rows)

// ---------------------------------------------------------------------
// CPU reference: plain triple loop. C = A · B.
// ---------------------------------------------------------------------
void spmmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) C[i * K + k] = 0.0f;
        for (int n = 0; n < N; n++) {
            float a = A[i * N + n];
            if (a == 0.0f) continue;            // mirror the GPU zero-skip
            for (int k = 0; k < K; k++)
                C[i * K + k] += a * B[n * K + k];
        }
    }
}

// ---------------------------------------------------------------------
// Strategy 1 — dense naive GEMM. One thread per C element, no sparsity.
// Baseline to measure how much the zero-skip actually buys us.
// ---------------------------------------------------------------------
__global__ void spmmDenseNaive(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // over K
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // over M
    if (row >= M || col >= K) return;

    float acc = 0.0f;
    for (int n = 0; n < N; n++)
        acc += A[row * N + n] * B[n * K + col];        // multiplies even when A==0
    C[row * K + col] = acc;
}

// ---------------------------------------------------------------------
// Strategy 2 — zero-skip dense (the one `solve` uses).
//
// Thread (tx,ty): col = ...x  (over K),  row = ...y  (over M).
// A warp = 32 consecutive tx, SAME ty  ->  same `row`, consecutive `col`.
//
//   A[row*N + n]      : row,n are warp-uniform -> all 32 lanes read the
//                       SAME address. Broadcast load, served from L1/L2.
//   if (a != 0.0f)    : condition depends only on (row,n) -> warp-uniform.
//                       The ENTIRE warp skips together. Zero divergence.
//   B[n*K + col]      : consecutive col -> consecutive addresses ->
//                       one coalesced 128-byte transaction.
//
// With ~65% zeros, ~65% of the n-iterations skip both the coalesced B
// read and the FMA, for the whole warp. `nnz` is not needed here.
// ---------------------------------------------------------------------
__global__ void spmmZeroSkip(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // over K
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // over M
    if (row >= M || col >= K) return;

    float acc = 0.0f;
    const float* Arow = A + (size_t)row * N;
    for (int n = 0; n < N; n++) {
        float a = Arow[n];                 // warp-uniform broadcast load
        if (a != 0.0f)                     // warp-uniform branch (no divergence)
            acc += a * B[(size_t)n * K + col];   // coalesced over col
    }
    C[(size_t)row * K + col] = acc;
}

// ---------------------------------------------------------------------
// Strategy 3 — convert dense A to CSR on the GPU, then SpMM from CSR.
//
//   csrRowCountKernel : thread per row -> count nonzeros of that row
//   exclusiveScanKernel: rowPtr[0]=0, rowPtr[r+1]=rowPtr[r]+count[r]
//                        (serial single-thread scan — see docs: this O(M)
//                         scan is the conversion bottleneck; a real impl
//                         would use the parallel Blelloch scan kernel)
//   csrFillKernel     : thread per row -> scatter (col,val) into CSR
//   spmmCSRKernel     : C[r][k] = Σ vals[i]·B[colIdx[i]][k]  over row r
// ---------------------------------------------------------------------
__global__ void csrRowCountKernel(const float* A, int* rowCount,
                                   int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const float* Arow = A + (size_t)row * N;
    int cnt = 0;
    for (int n = 0; n < N; n++) cnt += (Arow[n] != 0.0f);
    rowCount[row] = cnt;
}

__global__ void exclusiveScanKernel(const int* rowCount, int* rowPtr, int M) {
    // Single thread, serial prefix sum over M entries. Correct and simple;
    // intentionally the slow part of CSR conversion (highlighted in docs).
    int acc = 0;
    rowPtr[0] = 0;
    for (int r = 0; r < M; r++) {
        acc += rowCount[r];
        rowPtr[r + 1] = acc;
    }
}

__global__ void csrFillKernel(const float* A, const int* rowPtr,
                               int* colIdx, float* vals, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const float* Arow = A + (size_t)row * N;
    int pos = rowPtr[row];
    for (int n = 0; n < N; n++) {
        float a = Arow[n];
        if (a != 0.0f) { colIdx[pos] = n; vals[pos] = a; pos++; }
    }
}

__global__ void spmmCSRKernel(const int* rowPtr, const int* colIdx,
                              const float* vals, const float* B, float* C,
                              int M, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // over K
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // over M
    if (row >= M || col >= K) return;

    int start = rowPtr[row], end = rowPtr[row + 1];
    float acc = 0.0f;
    for (int i = start; i < end; i++)                  // only nnz of this row
        acc += vals[i] * B[(size_t)colIdx[i] * K + col];   // coalesced over col
    C[(size_t)row * K + col] = acc;
}

// ---------------------------------------------------------------------
// REQUIRED ENTRY POINT — signature must not change.
// A, B, C are device pointers. Result is written to C.
// Uses strategy 2 (zero-skip): fastest no-conversion path, no extra
// memory, divergence-free. `nnz` is unused by this strategy.
// ---------------------------------------------------------------------
void solve(const float* A, const float* B, float* C,
           int M, int N, int K, int nnz) {
    (void)nnz;  // not needed by the zero-skip strategy
    dim3 block(TILE, TILE);
    dim3 grid(cdiv(K, TILE), cdiv(M, TILE));
    spmmZeroSkip<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------
// Benchmark harness (same pattern as the other kernels in this repo).
// ---------------------------------------------------------------------
enum Mode { MODE_ZEROSKIP, MODE_NAIVE, MODE_CSR };

// Returns kernel time in microseconds. For CSR, `buildUs` (out) gets the
// dense->CSR conversion time, reported separately from the multiply.
float launchSpmm(Mode mode, const float* d_A, const float* d_B, float* d_C,
                 int M, int N, int K,
                 int* d_rowCount, int* d_rowPtr, int* d_colIdx, float* d_vals,
                 bool warmup, double* buildUs = nullptr) {
    dim3 block(TILE, TILE);
    dim3 grid(cdiv(K, TILE), cdiv(M, TILE));
    int rowBlk = 256, rowGrid = cdiv(M, rowBlk);

    auto buildCSR = [&]() {
        csrRowCountKernel<<<rowGrid, rowBlk>>>(d_A, d_rowCount, M, N);
        exclusiveScanKernel<<<1, 1>>>(d_rowCount, d_rowPtr, M);
        csrFillKernel<<<rowGrid, rowBlk>>>(d_A, d_rowPtr, d_colIdx, d_vals, M, N);
    };

    if (warmup) {
        if (mode == MODE_NAIVE)
            spmmDenseNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        else if (mode == MODE_ZEROSKIP)
            spmmZeroSkip<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        else { buildCSR(); spmmCSRKernel<<<grid, block>>>(d_rowPtr, d_colIdx,
                                                          d_vals, d_B, d_C, M, K); }
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return 0.0f;
    }

    cudaEvent_t s, e;
    CHECK_CUDA_ERROR(cudaEventCreate(&s));
    CHECK_CUDA_ERROR(cudaEventCreate(&e));

    if (mode == MODE_CSR && buildUs) {
        // Time the conversion separately so the trade-off is visible.
        cudaEvent_t bs, be;
        CHECK_CUDA_ERROR(cudaEventCreate(&bs));
        CHECK_CUDA_ERROR(cudaEventCreate(&be));
        CHECK_CUDA_ERROR(cudaEventRecord(bs));
        buildCSR();
        CHECK_CUDA_ERROR(cudaEventRecord(be));
        CHECK_CUDA_ERROR(cudaEventSynchronize(be));
        float bms; CHECK_CUDA_ERROR(cudaEventElapsedTime(&bms, bs, be));
        *buildUs = bms * 1000.0;
        CHECK_CUDA_ERROR(cudaEventDestroy(bs));
        CHECK_CUDA_ERROR(cudaEventDestroy(be));
    }

    CHECK_CUDA_ERROR(cudaEventRecord(s));
    if (mode == MODE_NAIVE)
        spmmDenseNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    else if (mode == MODE_ZEROSKIP)
        spmmZeroSkip<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    else
        spmmCSRKernel<<<grid, block>>>(d_rowPtr, d_colIdx, d_vals, d_B, d_C, M, K);
    CHECK_CUDA_ERROR(cudaEventRecord(e));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventSynchronize(e));
    float ms; CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, s, e));
    CHECK_CUDA_ERROR(cudaEventDestroy(s));
    CHECK_CUDA_ERROR(cudaEventDestroy(e));
    return ms * 1000.0f;   // microseconds
}

bool compareResults(const float* ref, const float* got, int n) {
    for (int i = 0; i < n; i++) {
        float tol = 1e-3f * (1.0f + fabsf(ref[i]));   // relative+abs (GEMM sums)
        if (fabsf(ref[i] - got[i]) > tol) {
            printf("Mismatch at %d: CPU = %f, GPU = %f\n", i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}

// Fill A with ~`sparsity` fraction of zeros; nonzeros are in [0.1,10].
// Returns the exact nnz so the value matches what solve() would receive.
int initSparseA(float* A, int M, int N, float sparsity) {
    int nnz = 0;
    for (int i = 0; i < M * N; i++) {
        float u = (float)rand() / RAND_MAX;
        if (u < sparsity) { A[i] = 0.0f; }
        else { A[i] = 0.1f + ((float)rand() / RAND_MAX) * 9.9f; nnz++; }
    }
    return nnz;
}

void initDense(float* B, int n) {
    for (int i = 0; i < n; i++) B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    float sparsity = 0.65f;            // 65% zeros (within the 60-70% band)
    Mode mode = MODE_ZEROSKIP;
    bool compareAll = false;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) sparsity = atof(argv[4]);
    if (argc > 5) {
        if (!strcmp(argv[5], "naive"))    mode = MODE_NAIVE;
        else if (!strcmp(argv[5], "csr")) mode = MODE_CSR;
        else if (!strcmp(argv[5], "compare")) compareAll = true;
    }
    if (M <= 0 || N <= 0 || K <= 0 || sparsity < 0.0f || sparsity >= 1.0f) {
        printf("Invalid args. Using M=N=K=512, sparsity=0.65.\n");
        M = N = K = 512; sparsity = 0.65f;
    }

    printf("Sparse×Dense MatMul — A[%d×%d] (sparse) · B[%d×%d] -> C[%d×%d]\n",
           M, N, K, N, K, M, K);
    printf("Target sparsity: %.0f%% zeros\n", sparsity * 100.0f);

    size_t aN = (size_t)M * N, bN = (size_t)N * K, cN = (size_t)M * K;
    float* h_A     = (float*)malloc(aN * sizeof(float));
    float* h_B     = (float*)malloc(bN * sizeof(float));
    float* h_C_cpu = (float*)malloc(cN * sizeof(float));
    float* h_C_gpu = (float*)malloc(cN * sizeof(float));
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Host allocation failed\n"); return 1;
    }

    srand(time(NULL));
    int nnz = initSparseA(h_A, M, N, sparsity);
    initDense(h_B, N * K);
    printf("Actual nnz: %d / %d  (%.1f%% zeros)\n",
           nnz, M * N, 100.0 * (1.0 - (double)nnz / (M * N)));

    // CPU reference + timing
    const int cpuTrials = 5;
    printf("CPU reference (%d trials)...\n", cpuTrials);
    double cpuUs = 0.0;
    for (int t = 0; t < cpuTrials; t++) {
        auto a = std::chrono::high_resolution_clock::now();
        spmmCPU(h_A, h_B, h_C_cpu, M, N, K);
        auto b = std::chrono::high_resolution_clock::now();
        cpuUs += std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    }
    cpuUs /= cpuTrials;
    printf("CPU Time (avg of %d): %.2f us\n", cpuTrials, cpuUs);

    // Device memory
    float *d_A, *d_B, *d_C, *d_vals;
    int   *d_rowCount, *d_rowPtr, *d_colIdx;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, aN * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bN * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, cN * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_rowCount, M * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_rowPtr, (M + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_colIdx, (size_t)nnz * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_vals,   (size_t)nnz * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, aN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bN * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = 10, trials = 100;

    auto runOne = [&](Mode m, const char* name) {
        for (int t = 0; t < warmup; t++)
            launchSpmm(m, d_A, d_B, d_C, M, N, K,
                       d_rowCount, d_rowPtr, d_colIdx, d_vals, true);
        double us = 0.0, bAcc = 0.0;
        for (int t = 0; t < trials; t++) {
            double buildUs = 0.0;
            us += launchSpmm(m, d_A, d_B, d_C, M, N, K,
                             d_rowCount, d_rowPtr, d_colIdx, d_vals, false,
                             (m == MODE_CSR ? &buildUs : nullptr));
            bAcc += buildUs;
        }
        us /= trials; bAcc /= trials;
        CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, cN * sizeof(float),
                                    cudaMemcpyDeviceToHost));
        bool ok = compareResults(h_C_cpu, h_C_gpu, M * K);
        if (m == MODE_CSR)
            printf("  %-10s  multiply %.2f us | CSR build %.2f us | total %.2f us  [%s]\n",
                   name, us, bAcc, us + bAcc, ok ? "OK" : "WRONG");
        else
            printf("  %-10s  %.2f us  (%.2fx vs CPU)  [%s]\n",
                   name, us, cpuUs / us, ok ? "OK" : "WRONG");
        return ok;
    };

    printf("\nGPU (%d warmup, %d timed trials each):\n", warmup, trials);
    if (compareAll) {
        runOne(MODE_NAIVE,    "naive");
        runOne(MODE_ZEROSKIP, "zeroskip");
        runOne(MODE_CSR,      "csr");
        printf("\nzeroskip is what solve() uses: no conversion, no extra "
               "memory, divergence-free.\n");
    } else {
        const char* nm = (mode == MODE_NAIVE) ? "naive"
                       : (mode == MODE_CSR)   ? "csr" : "zeroskip";
        bool ok = runOne(mode, nm);
        if (ok) {
            printf("\nSample C[0..2][0..2]:\n");
            for (int i = 0; i < 3 && i < M; i++) {
                for (int k = 0; k < 3 && k < K; k++)
                    printf("%9.3f ", h_C_gpu[i * K + k]);
                printf("\n");
            }
        }
    }

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_rowCount); cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_vals);
    return 0;
}
