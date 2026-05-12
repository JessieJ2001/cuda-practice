# CUDA Programming Practice

A small collection of CUDA kernels for learning GPU programming from the ground up. Each kernel ships with a CPU reference implementation, a benchmark harness, and an in-depth walkthrough in [`docs/`](docs/index.html).

## Project Structure

```
.
├── vector_addition.cu     # 1. baseline kernel — warp execution, perfect coalescing
├── transpose.cu           # 2. coalescing + shared-memory tiling + bank conflicts
├── softmax.cu             # 3. tree reductions + atomicAdd across CTAs
├── topk.cu                # 4. radix-bucket selection (no sort)
├── attention.cu           # 5. naive 3-kernel scaled dot-product attention
├── flash_attention.cu     # 6. fused single-kernel attention with online softmax
├── common.h               # CHECK_CUDA_ERROR macro and cdiv helper
├── Makefile               # build + run targets
├── docs/                  # step-by-step HTML walkthroughs (open docs/index.html)
└── README.md              # this file
```

## Documentation

Open **[`docs/index.html`](docs/index.html)** in a browser for the full walkthrough — a beginner-friendly tour of every kernel with step-by-step animated diagrams, grounded in the GTX 1660 Super memory hierarchy. The docs cover:

- **GPU 101 primer** — memory hierarchy, SM, CTA, warp, coalescing, bank conflicts
- **One page per kernel** with annotated source, launch config, memory analysis, and animated SVG figures

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (`nvcc` in PATH)

The Makefile defaults to `-arch=sm_75` (Turing — GTX 16xx and RTX 20xx). Update `CUDA_ARCH` in `Makefile` for other GPUs:

| GPU family    | `-arch` |
| ------------- | ------- |
| GTX 900       | sm_52   |
| GTX 10xx      | sm_61   |
| GTX 16xx, RTX 20xx | sm_75 |
| RTX 30xx      | sm_86   |
| RTX 40xx      | sm_89   |

## Quick Start

```bash
# Build everything
make

# Run individual kernels (defaults)
make run-vector             # vector add, 1M elements
make run-transpose          # 1024×1024 transpose
make run-softmax            # softmax on 10K elements
make run-topk               # top-10 of 10K
make run-attention          # Q[512,64], K/V[512,64]
make run-flash-attention    # same dims, fused kernel

# Compare naive vs optimized
make run-transpose-compare    # naive vs shared-memory transpose
make run-attention-compare    # naive 3-kernel vs flash attention
```

Each program also accepts CLI args for custom sizes — see `make help` for the full list.

## The Kernels

### 1. Vector Addition · [`vector_addition.cu`](vector_addition.cu) · [docs →](docs/vector_addition.html)

`C[i] = A[i] + B[i]`. The simplest possible CUDA kernel. Used as a baseline to introduce warps, lockstep execution, and perfect coalescing.

```bash
make run-vector-small       # 1K elements
make run-vector-large       # 10M elements
```

### 2. Matrix Transpose · [`transpose.cu`](transpose.cu) · [docs →](docs/transpose.html)

Two kernels side-by-side: a naive one that wastes ~32× DRAM bandwidth on the strided write, and a shared-memory version with the `(TILE_SIZE+1)` padding trick to dodge bank conflicts.

```bash
make run-transpose-compare  # naive vs shared-memory comparison
```

### 3. Softmax · [`softmax.cu`](softmax.cu) · [docs →](docs/softmax.html)

Three-kernel pipeline (find-max → exp+sum → normalize) demonstrating tree reductions in shared memory, atomic aggregation across CTAs, and the numerical-stability max-trick.

```bash
make run-softmax-test       # runs on [1, 2, 3]
make run-softmax-large      # 500K elements
```

### 4. Top-K Selection · [`topk.cu`](topk.cu) · [docs →](docs/topk.html)

Radix-bucket selection: bin floats by their top bits in a shared-memory histogram, walk the histogram on the host to find which bucket contains the K-th element, repeat until the threshold is pinned down. Then one filter pass collects all elements ≥ threshold.

```bash
make run-topk-test          # 100 elements, K=5, known pattern
make run-topk-ratios        # different K/N ratios
```

### 5. Attention (naive) · [`attention.cu`](attention.cu) · [docs →](docs/attention.html)

Scaled dot-product attention via three kernel launches:
1. `attentionScoresKernel`: scores = (Q · Kᵀ) / √d, materializes the full `[M × N]` matrix.
2. `softmaxRowsKernel`: row-wise softmax in place, one CTA per row.
3. `attentionOutputKernel`: O = P · V.

```bash
make run-attention          # Q[512,64], K/V[512,64]
make run-attention-large    # Q[1024,128], K/V[1024,128]
```

### 6. Flash Attention · [`flash_attention.cu`](flash_attention.cu) · [docs →](docs/flash_attention.html)

The same math fused into a **single kernel** that never materializes the `[M × N]` scores buffer. Each CTA streams over keys while maintaining a running max, running denominator, and running output accumulator — the online-softmax trick from Dao et al.

```bash
make run-flash-attention            # same dims as naive attention
make run-attention-compare          # side-by-side benchmark
```

## Common Mistakes to Avoid

- **Forgetting bounds checks** — extra threads launched for the tail of a grid must be guarded with `if (idx < N)`.
- **Missing `__syncthreads()`** between shared-memory write and read phases — random output.
- **Conditional `__syncthreads()`** — barrier must be hit by every thread in the CTA uniformly, or you deadlock.
- **Numerical instability** — always subtract the row max before `exp()` in softmax-style code.
- **Forgetting `cudaMemset`** on global atomic accumulators between trials — old values leak in.
- **Architecture mismatch** — update `CUDA_ARCH` in the Makefile for your GPU.

## Make Targets

`make help` lists every target. The categories are:

- **build:** `all`, `vector_add`, `transpose`, `softmax`, `topk`, `attention`, `flash_attention`
- **vector:** `run-vector`, `run-vector-{small,medium,large,custom}`
- **transpose:** `run-transpose`, `run-transpose-{test,small,medium,large,custom,naive,compare}`
- **softmax:** `run-softmax`, `run-softmax-{test,small,medium,large,custom}`
- **topk:** `run-topk`, `run-topk-{test,small,medium,large,custom,compare,ratios}`
- **attention:** `run-attention`, `run-attention-{small,medium,large,custom,compare}`
- **flash attention:** `run-flash-attention`, `run-flash-attention-{small,medium,large,custom}`
- **utilities:** `clean`, `gpu-info`, `cuda-check`, `help`

## Suggested Learning Path

1. **Vector Addition** — understand warps, blocks, and coalescing.
2. **Transpose** — learn shared memory and bank conflicts.
3. **Softmax** — learn parallel reductions and the multi-kernel pattern.
4. **Top-K** — learn shared-memory histograms and grid-stride loops.
5. **Attention** — combine everything into a 3-kernel pipeline.
6. **Flash Attention** — fuse the pipeline and stream through the online-softmax trick.

Walk through each kernel's `docs/<name>.html` page alongside the source. The pages cite real line numbers from the `.cu` files.
