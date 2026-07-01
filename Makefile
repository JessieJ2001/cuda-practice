# Makefile for CUDA Programming Practice

# Compiler and flags
NVCC = nvcc
CFLAGS = -O2 -std=c++17
CUDA_ARCH = -arch=sm_75  # Change this based on your GPU architecture

# Target executables
VECTOR_TARGET = vector_add
VECTOR_SOURCE = vector_addition.cu
SOFTMAX_TARGET = softmax
SOFTMAX_SOURCE = softmax.cu
CONV2D_TARGET = conv2d
CONV2D_SOURCE = 2d_convolution.cu
TRANSPOSE_TARGET = transpose
TRANSPOSE_SOURCE = transpose.cu
SCAN_TARGET = scan
SCAN_SOURCE = scan.cu
TOPK_TARGET = topk
TOPK_SOURCE = topk.cu
ATTENTION_TARGET = attention
ATTENTION_SOURCE = attention.cu
FLASH_ATTENTION_TARGET = flash_attention
FLASH_ATTENTION_SOURCE = flash_attention.cu
SPMM_TARGET = spmm
SPMM_SOURCE = spmm.cu

# Default target - build all programs
all: $(VECTOR_TARGET) $(SOFTMAX_TARGET) $(CONV2D_TARGET) $(TRANSPOSE_TARGET) $(SCAN_TARGET) $(TOPK_TARGET) $(ATTENTION_TARGET) $(FLASH_ATTENTION_TARGET) $(SPMM_TARGET)

# Compile the vector addition program
$(VECTOR_TARGET): $(VECTOR_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(VECTOR_TARGET) $(VECTOR_SOURCE)

# Compile the softmax program
$(SOFTMAX_TARGET): $(SOFTMAX_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(SOFTMAX_TARGET) $(SOFTMAX_SOURCE)

# Compile the 2D convolution program
$(CONV2D_TARGET): $(CONV2D_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(CONV2D_TARGET) $(CONV2D_SOURCE)

# Compile the transpose program
$(TRANSPOSE_TARGET): $(TRANSPOSE_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(TRANSPOSE_TARGET) $(TRANSPOSE_SOURCE)

# Compile the scan program
$(SCAN_TARGET): $(SCAN_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(SCAN_TARGET) $(SCAN_SOURCE)

# Compile the topk program
$(TOPK_TARGET): $(TOPK_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(TOPK_TARGET) $(TOPK_SOURCE)

# Compile the attention program
$(ATTENTION_TARGET): $(ATTENTION_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(ATTENTION_TARGET) $(ATTENTION_SOURCE)

# Compile the flash attention program
$(FLASH_ATTENTION_TARGET): $(FLASH_ATTENTION_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(FLASH_ATTENTION_TARGET) $(FLASH_ATTENTION_SOURCE)

# Compile the sparse×dense matmul program
$(SPMM_TARGET): $(SPMM_SOURCE)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $(SPMM_TARGET) $(SPMM_SOURCE)

# Vector Addition targets
# Run with default parameters (1M elements)
run-vector: $(VECTOR_TARGET)
	./$(VECTOR_TARGET)

# Run with custom vector size (e.g., make run-vector-custom N=1000)
run-vector-custom: $(VECTOR_TARGET)
	./$(VECTOR_TARGET) $(N)

# Run with small vector for testing (1000 elements)
run-vector-small: $(VECTOR_TARGET)
	./$(VECTOR_TARGET) 1000

# Run with medium vector (100K elements)
run-vector-medium: $(VECTOR_TARGET)
	./$(VECTOR_TARGET) 100000

# Run with large vector (10M elements)
run-vector-large: $(VECTOR_TARGET)
	./$(VECTOR_TARGET) 10000000

# Softmax targets
# Run with default parameters (10K elements)
run-softmax: $(SOFTMAX_TARGET)
	./$(SOFTMAX_TARGET)

# Run with test case values
run-softmax-test: $(SOFTMAX_TARGET)
	./$(SOFTMAX_TARGET) 3 test

# Run with custom size (e.g., make run-softmax-custom N=5000)
run-softmax-custom: $(SOFTMAX_TARGET)
	./$(SOFTMAX_TARGET) $(N)

# Run with small array for testing (100 elements)
run-softmax-small: $(SOFTMAX_TARGET)
	./$(SOFTMAX_TARGET) 100

# Run with medium array (50K elements)
run-softmax-medium: $(SOFTMAX_TARGET)
	./$(SOFTMAX_TARGET) 50000

# Run with large array (500K elements)
run-softmax-large: $(SOFTMAX_TARGET)
	./$(SOFTMAX_TARGET) 500000

# 2D Convolution targets
# Run with default parameters (64x64 input, 3x3 kernel)
run-conv2d: $(CONV2D_TARGET)
	./$(CONV2D_TARGET)

# Run with small test case (8x8 input for visual debugging)
run-conv2d-small: $(CONV2D_TARGET)
	./$(CONV2D_TARGET) 8 8 3 3 test edge

# Run with different kernel types
run-conv2d-blur: $(CONV2D_TARGET)
	./$(CONV2D_TARGET) 32 32 3 3 test blur

run-conv2d-identity: $(CONV2D_TARGET)
	./$(CONV2D_TARGET) 32 32 3 3 test identity

# Run with custom size (e.g., make run-conv2d-custom ROWS=128 COLS=128)
run-conv2d-custom: $(CONV2D_TARGET)
	./$(CONV2D_TARGET) $(ROWS) $(COLS) 3 3

# Run with large input (256x256)
run-conv2d-large: $(CONV2D_TARGET)
	./$(CONV2D_TARGET) 256 256 3 3

# Transpose targets
# Run with default parameters (1024x1024 matrix)
run-transpose: $(TRANSPOSE_TARGET)
	./$(TRANSPOSE_TARGET)

# Run with test pattern for verification (8x8 matrix)
run-transpose-test: $(TRANSPOSE_TARGET)
	./$(TRANSPOSE_TARGET) 8 8 pattern

# Run with small test case (16x16 matrix)
run-transpose-small: $(TRANSPOSE_TARGET)
	./$(TRANSPOSE_TARGET) 16 16

# Run with medium test case (256x256 matrix)
run-transpose-medium: $(TRANSPOSE_TARGET)
	./$(TRANSPOSE_TARGET) 256 256

# Run with large test case (1024x1024 matrix)
run-transpose-large: $(TRANSPOSE_TARGET)
	./$(TRANSPOSE_TARGET) 1024 1024

# Run with custom size (e.g., make run-transpose-custom ROWS=128 COLS=64)
run-transpose-custom: $(TRANSPOSE_TARGET)
	./$(TRANSPOSE_TARGET) $(ROWS) $(COLS)

# Run with naive implementation (no shared memory)
run-transpose-naive: $(TRANSPOSE_TARGET)
	./$(TRANSPOSE_TARGET) 256 256 pattern naive

# Compare naive vs shared memory implementations
run-transpose-compare: $(TRANSPOSE_TARGET)
	@echo "Running naive implementation:"
	./$(TRANSPOSE_TARGET) 256 256 pattern naive
	@echo ""
	@echo "Running shared memory implementation:"
	./$(TRANSPOSE_TARGET) 256 256 pattern

# Scan targets
# Run with default parameters (10K elements, exclusive scan)
run-scan: $(SCAN_TARGET)
	./$(SCAN_TARGET)

# Run with test pattern (all ones for easy verification)
run-scan-ones: $(SCAN_TARGET)
	./$(SCAN_TARGET) 20 ones

# Run with small array for testing (100 elements)
run-scan-small: $(SCAN_TARGET)
	./$(SCAN_TARGET) 100 ones

# Run with medium array (50K elements)
run-scan-medium: $(SCAN_TARGET)
	./$(SCAN_TARGET) 50000

# Run with large array (1M elements)
run-scan-large: $(SCAN_TARGET)
	./$(SCAN_TARGET) 1000000

# Run with custom size (e.g., make run-scan-custom N=5000)
run-scan-custom: $(SCAN_TARGET)
	./$(SCAN_TARGET) $(N)

# Run inclusive scan instead of exclusive
run-scan-inclusive: $(SCAN_TARGET)
	./$(SCAN_TARGET) 100 ones inclusive

# Compare exclusive vs inclusive scan
run-scan-compare: $(SCAN_TARGET)
	@echo "Running exclusive scan:"
	./$(SCAN_TARGET) 20 ones
	@echo ""
	@echo "Running inclusive scan:"
	./$(SCAN_TARGET) 20 ones inclusive

# TopK targets
# Run with default parameters (10K elements, K=10)
run-topk: $(TOPK_TARGET)
	./$(TOPK_TARGET)

# Run with test pattern (known top values)
run-topk-test: $(TOPK_TARGET)
	./$(TOPK_TARGET) 100 5 pattern

# Run with small test case (100 elements, K=5)
run-topk-small: $(TOPK_TARGET)
	./$(TOPK_TARGET) 100 5

# Run with medium test case (50K elements, K=10)
run-topk-medium: $(TOPK_TARGET)
	./$(TOPK_TARGET) 50000 10

# Run with large test case (1M elements, K=100)
run-topk-large: $(TOPK_TARGET)
	./$(TOPK_TARGET) 1000000 100

# Run with custom size and K (e.g., make run-topk-custom N=1000 K=50)
run-topk-custom: $(TOPK_TARGET)
	./$(TOPK_TARGET) $(N) $(K)

# Run with heap-based implementation
run-topk-heap: $(TOPK_TARGET)
	./$(TOPK_TARGET) 10000 20 pattern heap

# Compare sorting vs heap implementations
run-topk-compare: $(TOPK_TARGET)
	@echo "Running sorting-based implementation:"
	./$(TOPK_TARGET) 10000 20 pattern
	@echo ""
	@echo "Running heap-based implementation:"
	./$(TOPK_TARGET) 10000 20 pattern heap

# Test different K/N ratios
run-topk-ratios: $(TOPK_TARGET)
	@echo "Testing small K (K=5, N=10000):"
	./$(TOPK_TARGET) 10000 5
	@echo ""
	@echo "Testing medium K (K=100, N=10000):"
	./$(TOPK_TARGET) 10000 100
	@echo ""
	@echo "Testing large K (K=1000, N=10000):"
	./$(TOPK_TARGET) 10000 1000

# Attention targets
# Run with default parameters (Q[512,64], K/V[512,64])
run-attention: $(ATTENTION_TARGET)
	./$(ATTENTION_TARGET)

# Run with small test case (Q[64,32], K/V[64,32])
run-attention-small: $(ATTENTION_TARGET)
	./$(ATTENTION_TARGET) 64 64 32

# Run with medium test case (Q[256,64], K/V[256,64])
run-attention-medium: $(ATTENTION_TARGET)
	./$(ATTENTION_TARGET) 256 256 64

# Run with large test case (Q[1024,128], K/V[1024,128])
run-attention-large: $(ATTENTION_TARGET)
	./$(ATTENTION_TARGET) 1024 1024 128

# Run with custom dimensions (e.g., make run-attention-custom M=128 N=256 D=64)
run-attention-custom: $(ATTENTION_TARGET)
	./$(ATTENTION_TARGET) $(M) $(N) $(D)

# Flash Attention targets (single fused streaming-softmax kernel)
# Run with default parameters (Q[512,64], K/V[512,64])
run-flash-attention: $(FLASH_ATTENTION_TARGET)
	./$(FLASH_ATTENTION_TARGET)

# Run with small test case (Q[64,32], K/V[64,32])
run-flash-attention-small: $(FLASH_ATTENTION_TARGET)
	./$(FLASH_ATTENTION_TARGET) 64 64 32

# Run with medium test case (Q[256,64], K/V[256,64])
run-flash-attention-medium: $(FLASH_ATTENTION_TARGET)
	./$(FLASH_ATTENTION_TARGET) 256 256 64

# Run with large test case (Q[1024,128], K/V[1024,128])
run-flash-attention-large: $(FLASH_ATTENTION_TARGET)
	./$(FLASH_ATTENTION_TARGET) 1024 1024 128

# Run with custom dimensions (e.g., make run-flash-attention-custom M=128 N=256 D=64)
run-flash-attention-custom: $(FLASH_ATTENTION_TARGET)
	./$(FLASH_ATTENTION_TARGET) $(M) $(N) $(D)

# Compare the materialized-scores attention vs. the fused flash-attention kernel
run-attention-compare: $(ATTENTION_TARGET) $(FLASH_ATTENTION_TARGET)
	@echo "Running attention.cu (3 kernels, materializes [M,N] scores):"
	./$(ATTENTION_TARGET) 1024 1024 128
	@echo ""
	@echo "Running flash_attention.cu (1 fused kernel, streaming softmax):"
	./$(FLASH_ATTENTION_TARGET) 1024 1024 128

# Sparse×Dense MatMul targets (A sparse M×N · B dense N×K -> C M×K)
# Run with default parameters (512×512×512, 65% zeros, zero-skip kernel)
run-spmm: $(SPMM_TARGET)
	./$(SPMM_TARGET)

# Small test case (64×64×64)
run-spmm-small: $(SPMM_TARGET)
	./$(SPMM_TARGET) 64 64 64 0.65

# Medium test case (256×256×256)
run-spmm-medium: $(SPMM_TARGET)
	./$(SPMM_TARGET) 256 256 256 0.65

# Large test case (1024×1024×1024)
run-spmm-large: $(SPMM_TARGET)
	./$(SPMM_TARGET) 1024 1024 1024 0.65

# Custom dims (e.g., make run-spmm-custom M=512 N=512 K=512 S=0.7)
run-spmm-custom: $(SPMM_TARGET)
	./$(SPMM_TARGET) $(M) $(N) $(K) $(S)

# Naive dense baseline (sparsity ignored)
run-spmm-naive: $(SPMM_TARGET)
	./$(SPMM_TARGET) 512 512 512 0.65 naive

# CSR-conversion path (reports build vs multiply time separately)
run-spmm-csr: $(SPMM_TARGET)
	./$(SPMM_TARGET) 512 512 512 0.65 csr

# Compare all three strategies side by side
run-spmm-compare: $(SPMM_TARGET)
	./$(SPMM_TARGET) 512 512 512 0.65 compare

# Backward compatibility - run vector addition by default
run: run-vector
run-custom: run-vector-custom
run-small: run-vector-small
run-medium: run-vector-medium
run-large: run-vector-large

# Clean build artifacts
clean:
	rm -f $(VECTOR_TARGET) $(SOFTMAX_TARGET) $(CONV2D_TARGET) $(TRANSPOSE_TARGET) $(SCAN_TARGET) $(TOPK_TARGET) $(ATTENTION_TARGET) $(FLASH_ATTENTION_TARGET) $(SPMM_TARGET)

# Show GPU information
gpu-info:
	nvidia-smi

# Check CUDA installation
cuda-check:
	nvcc --version

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  all                    - Build all programs (default)"
	@echo "  vector_add             - Build vector addition program"
	@echo "  softmax                - Build softmax program"
	@echo "  conv2d                 - Build 2D convolution program"
	@echo "  transpose              - Build transpose program"
	@echo "  scan                   - Build scan program"
	@echo "  topk                   - Build topk program"
	@echo "  attention              - Build attention program"
	@echo "  flash_attention        - Build flash attention program (fused kernel)"
	@echo ""
	@echo "Vector Addition (backward compatible):"
	@echo "  run                    - Run vector addition with 1M elements"
	@echo "  run-small              - Run with 1K elements"
	@echo "  run-medium             - Run with 100K elements"
	@echo "  run-large              - Run with 10M elements"
	@echo "  run-custom N=size      - Run with custom size"
	@echo ""
	@echo "Vector Addition (explicit):"
	@echo "  run-vector             - Run vector addition with 1M elements"
	@echo "  run-vector-small       - Run with 1K elements"
	@echo "  run-vector-medium      - Run with 100K elements"
	@echo "  run-vector-large       - Run with 10M elements"
	@echo "  run-vector-custom N=size - Run with custom size"
	@echo ""
	@echo "Softmax:"
	@echo "  run-softmax            - Run softmax with 10K elements"
	@echo "  run-softmax-test       - Run with test case [1,2,3]"
	@echo "  run-softmax-small      - Run with 100 elements"
	@echo "  run-softmax-medium     - Run with 50K elements"
	@echo "  run-softmax-large      - Run with 500K elements"
	@echo "  run-softmax-custom N=size - Run with custom size"
	@echo ""
	@echo "2D Convolution:"
	@echo "  run-conv2d             - Run with 64x64 input, 3x3 edge kernel"
	@echo "  run-conv2d-small       - Run with 8x8 input (visual debugging)"
	@echo "  run-conv2d-blur        - Run with 32x32 input, blur kernel"
	@echo "  run-conv2d-identity    - Run with 32x32 input, identity kernel"
	@echo "  run-conv2d-large       - Run with 256x256 input"
	@echo "  run-conv2d-custom ROWS=n COLS=m - Run with custom size"
	@echo ""
	@echo "Transpose:"
	@echo "  run-transpose          - Run transpose with 1024x1024 matrix"
	@echo "  run-transpose-test     - Run with 8x8 matrix (test pattern)"
	@echo "  run-transpose-small    - Run with 16x16 matrix"
	@echo "  run-transpose-medium   - Run with 256x256 matrix"
	@echo "  run-transpose-large    - Run with 1024x1024 matrix"
	@echo "  run-transpose-custom ROWS=n COLS=m - Run with custom size"
	@echo "  run-transpose-naive    - Run with 256x256 matrix (naive)"
	@echo "  run-transpose-compare  - Compare naive vs shared memory"
	@echo ""
	@echo "Scan:"
	@echo "  run-scan               - Run exclusive scan with 10K elements"
	@echo "  run-scan-ones          - Run exclusive scan with 20 ones"
	@echo "  run-scan-small         - Run exclusive scan with 100 ones"
	@echo "  run-scan-medium        - Run exclusive scan with 50K elements"
	@echo "  run-scan-large         - Run exclusive scan with 1M elements"
	@echo "  run-scan-custom N=size - Run with custom size"
	@echo "  run-scan-inclusive     - Run inclusive scan with 100 ones"
	@echo "  run-scan-compare       - Compare exclusive vs inclusive scan"
	@echo ""
	@echo "TopK:"
	@echo "  run-topk               - Run topk with 10K elements, K=10"
	@echo "  run-topk-test          - Run topk with 100 elements, K=5 (pattern)"
	@echo "  run-topk-small         - Run topk with 100 elements, K=5"
	@echo "  run-topk-medium        - Run topk with 50K elements, K=10"
	@echo "  run-topk-large         - Run topk with 1M elements, K=100"
	@echo "  run-topk-custom N=size K=k - Run with custom size and K"
	@echo "  run-topk-heap          - Run heap-based topk with 1000 elements, K=10 (pattern)"
	@echo "  run-topk-compare       - Compare sorting vs heap implementations"
	@echo "  run-topk-ratios        - Test different K/N ratios"
	@echo ""
	@echo "Attention (scaled dot-product):"
	@echo "  run-attention          - Run with Q[512,64], K/V[512,64]"
	@echo "  run-attention-small    - Run with Q[64,32], K/V[64,32]"
	@echo "  run-attention-medium   - Run with Q[256,64], K/V[256,64]"
	@echo "  run-attention-large    - Run with Q[1024,128], K/V[1024,128]"
	@echo "  run-attention-custom M=m N=n D=d - Run with custom dimensions"
	@echo ""
	@echo "Flash Attention (fused streaming-softmax kernel):"
	@echo "  run-flash-attention          - Run with Q[512,64], K/V[512,64]"
	@echo "  run-flash-attention-small    - Run with Q[64,32], K/V[64,32]"
	@echo "  run-flash-attention-medium   - Run with Q[256,64], K/V[256,64]"
	@echo "  run-flash-attention-large    - Run with Q[1024,128], K/V[1024,128]"
	@echo "  run-flash-attention-custom M=m N=n D=d - Run with custom dimensions"
	@echo "  run-attention-compare        - Compare materialized vs. fused attention"
	@echo ""
	@echo "Sparse×Dense MatMul (A sparse M×N · B dense N×K -> C M×K):"
	@echo "  run-spmm               - Run with 512×512×512, 65% zeros (zero-skip)"
	@echo "  run-spmm-small         - Run with 64×64×64"
	@echo "  run-spmm-medium        - Run with 256×256×256"
	@echo "  run-spmm-large         - Run with 1024×1024×1024"
	@echo "  run-spmm-custom M=.. N=.. K=.. S=.. - Run with custom dims/sparsity"
	@echo "  run-spmm-naive         - Naive dense baseline (sparsity ignored)"
	@echo "  run-spmm-csr           - CSR-conversion path (build vs multiply time)"
	@echo "  run-spmm-compare       - Compare naive vs zero-skip vs CSR"
	@echo ""
	@echo "Utilities:"
	@echo "  clean                  - Remove build artifacts"
	@echo "  gpu-info               - Show GPU information"
	@echo "  cuda-check             - Check CUDA installation"
	@echo "  help                   - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make run-vector-custom N=5000"
	@echo "  make run-softmax-custom N=1000"
	@echo "  make run-conv2d-custom ROWS=128 COLS=64"
	@echo "  make run-transpose-large"
	@echo "  make run-attention-custom M=128 N=256 D=64"
	@echo "  make run-flash-attention-large"
	@echo "  make run-spmm-compare"
	@echo "  make run-spmm-custom M=512 N=512 K=512 S=0.7"

.PHONY: all $(VECTOR_TARGET) $(SOFTMAX_TARGET) $(CONV2D_TARGET) $(TRANSPOSE_TARGET) $(SCAN_TARGET) $(TOPK_TARGET) $(ATTENTION_TARGET) $(FLASH_ATTENTION_TARGET) $(SPMM_TARGET) run run-custom run-small run-medium run-large \
        run-vector run-vector-custom run-vector-small run-vector-medium run-vector-large \
        run-softmax run-softmax-test run-softmax-custom run-softmax-small run-softmax-medium run-softmax-large \
        run-conv2d run-conv2d-small run-conv2d-blur run-conv2d-identity run-conv2d-custom run-conv2d-large \
        run-transpose run-transpose-test run-transpose-small run-transpose-medium run-transpose-large run-transpose-custom run-transpose-naive run-transpose-compare \
        run-scan run-scan-ones run-scan-small run-scan-medium run-scan-large run-scan-custom run-scan-inclusive run-scan-compare \
        run-topk run-topk-test run-topk-small run-topk-medium run-topk-large run-topk-custom run-topk-heap run-topk-compare run-topk-ratios \
        run-attention run-attention-small run-attention-medium run-attention-large run-attention-custom \
        run-flash-attention run-flash-attention-small run-flash-attention-medium run-flash-attention-large run-flash-attention-custom run-attention-compare \
        run-spmm run-spmm-small run-spmm-medium run-spmm-large run-spmm-custom run-spmm-naive run-spmm-csr run-spmm-compare \
        clean gpu-info cuda-check help