# req_1: Tensor Core Optimization with TF32

## Overview

Tensor Cores are specialized hardware units in NVIDIA GPUs (Volta architecture and later) designed to accelerate matrix multiplication operations. This optimization leverages Tensor Cores using TF32 (TensorFloat-32) precision to achieve massive speedups in matrix multiplication, which dominates the computational cost of GPT-2 inference.

## What Are Tensor Cores?

### Hardware Architecture

Tensor Cores are dedicated matrix multiplication units that can perform an entire 4×4×4 (or larger) matrix multiplication in a single instruction. On A40 GPUs:

```
Traditional CUDA Cores:
- One FMA (Fused Multiply-Add) per cycle per core
- 10,752 CUDA cores
- Peak FP32: 37.4 TFLOPS

Tensor Cores:
- One 16×16×8 matrix multiplication per operation
- 336 Tensor Cores (3rd generation)
- Peak TF32: 312 TFLOPS (8.3x faster!)
```

### TF32 Precision

TF32 is a hybrid precision format designed for AI workloads:

```
FP32 (Standard):   1 sign | 8 exponent | 23 mantissa = 32 bits
TF32 (Tensor):     1 sign | 8 exponent | 10 mantissa = 19 bits
FP16 (Half):       1 sign | 5 exponent | 10 mantissa = 16 bits

Key Properties:
- Same range as FP32 (8-bit exponent)
- Reduced precision (10 vs 23 mantissa bits)
- Automatic conversion from FP32
- Typically <0.1% accuracy loss for ML workloads
```

## Implementation Details

### WMMA API

The code uses CUDA's WMMA (Warp Matrix Multiply-Accumulate) API:

```cuda
#include <mma.h>
using namespace nvcuda;

// Tile dimensions for TF32: M × N × K
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
```

### Fragment Types

Fragments are warp-level abstractions for matrix tiles:

```cuda
// Matrix A fragment (input)
wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
               wmma::precision::tf32, wmma::row_major> a_frag;

// Matrix B fragment (weight)
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
               wmma::precision::tf32, wmma::col_major> b_frag;

// Accumulator fragment (output, FP32 precision)
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
```

**Key insight**: Accumulators remain FP32 for numerical stability, only the multiplication uses TF32.

### Kernel Structure

**File**: kernels_req_1/matmul.cuh:19-90

```cuda
__global__ void matmul_tensor_core_kernel(
    float *out, const float *inp, const float *weight,
    const float *bias, int C, int OC, int B, int T) {

    // Step 1: Calculate warp position
    int warpId = (threadIdx.x / WARP_SIZE);
    int warpRow = blockIdx.y * (blockDim.x / WARP_SIZE) + warpId;
    int warpCol = blockIdx.x;

    int bt = warpRow * WMMA_M;  // Row in output
    int oc = warpCol * WMMA_N;  // Column in output

    // Step 2: Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Step 3: Loop over K dimension in chunks of WMMA_K
    for (int k = 0; k < C; k += WMMA_K) {
        // Load 16×8 tile from input
        wmma::load_matrix_sync(a_frag, inp + bt * C + k, C);

        // Load 8×16 tile from weight (transposed)
        wmma::load_matrix_sync(b_frag, weight + oc * C + k, C);

        // Perform matrix multiplication using Tensor Cores
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Step 4: Store result
    wmma::store_matrix_sync(out + bt * OC + oc, acc_frag, OC,
                           wmma::mem_row_major);

    // Step 5: Add bias (if present)
    if (bias != NULL) {
        for (int i = 0; i < WMMA_M; i++) {
            for (int j = 0; j < WMMA_N; j++) {
                out[(bt + i) * OC + (oc + j)] += bias[oc + j];
            }
        }
    }
}
```

### Applied Kernels

#### 1. **Matrix Multiplication** (kernels_req_1/matmul.cuh)
- Main matmul kernel using Tensor Cores
- 128 threads per block (4 warps)
- Each warp computes one 16×16 output tile

#### 2. **Attention P@V** (kernels_req_1/attention.cuh:52-96)
- Custom P@V kernel: `pv_matmul_tensor_core_kernel`
- Dimensions: (T, T) @ (T, HS) → (T, HS)
- Same WMMA API usage pattern

## Why This Optimization Is Effective

### 1. **Massive Parallel Matrix Multiplication**

A single `mma_sync` instruction computes:
```
C[16×16] += A[16×16] @ B[16×8]
```

This represents:
- **2,048 FMA operations** (16 × 16 × 8 × 2)
- **Single instruction** (executed in ~1-2 cycles)
- **~1000x fewer instructions** than sequential code

### 2. **Hardware Acceleration**

Traditional CUDA cores:
```
for i in 0..16:
    for j in 0..16:
        for k in 0..8:
            C[i][j] += A[i][k] * B[k][j]  // 2048 instructions
```

Tensor Cores:
```
mma_sync(C, A, B, C);  // 1 instruction, 2048 operations
```

### 3. **Higher Throughput**

```
Theoretical Performance (A40 GPU):

FP32 CUDA Cores:
- 37.4 TFLOPS
- 10,752 cores × 2 ops/cycle × 1.74 GHz

TF32 Tensor Cores:
- 312 TFLOPS
- 336 Tensor Cores × 256 ops/cycle × 1.74 GHz

Speedup: 312 / 37.4 = 8.3x theoretical
Practical: 4-8x due to memory bottlenecks
```

### 4. **Better Energy Efficiency**

- Single Tensor Core operation = hundreds of CUDA core operations
- Lower power consumption per FLOP
- Less instruction fetch/decode overhead

## Performance Analysis

### Expected Speedup vs Baseline

| Matrix Size | Expected Speedup | Notes |
|-------------|------------------|-------|
| 64 × 64 × 64 | 3-4x | Small, alignment overhead |
| 256 × 256 × 256 | 5-7x | Good tile alignment |
| 1024 × 1024 × 1024 | 6-8x | Optimal for Tensor Cores |
| 2048 × 2048 × 2048 | 6-8x | Memory bound |

### vs req_0 (Register Tiling)

| Aspect | req_0 | req_1 | Advantage |
|--------|-------|-------|-----------|
| Speedup | 2-3x | 4-8x | req_1: 2-3x faster |
| Hardware | CUDA cores | Tensor Cores | req_1 |
| Precision | FP32 | TF32→FP32 | req_0 (slightly) |
| Complexity | High | Medium | req_1 |

### Key Profiling Metrics

#### Using Nsight Compute:

```bash
# Profile with Tensor Core metrics
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
              sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
              sm__inst_executed_pipe_tensor.sum,\
              smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
              gpu__time_duration.sum \
    -k matmul_tensor_core_kernel ./next_token_generation
```

**What to look for:**

1. **Tensor Pipe Utilization**
   - Metric: `sm__inst_executed_pipe_tensor.sum`
   - Expected: High value (indicates Tensor Core usage)
   - Goal: Most matmul operations go through Tensor pipeline

2. **SM Efficiency**
   - Metric: `sm__throughput.avg.pct_of_peak_sustained_elapsed`
   - Expected: 70-90%
   - Shows how well you're using the hardware

3. **Memory Efficiency**
   - Metric: `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   - Expected: 40-60% (becomes memory bound)
   - Higher than baseline despite faster compute

4. **Compute vs Memory Bound**
   ```
   SOL (Speed of Light):
   - Compute: Should be high (70-90%)
   - Memory: Medium (40-60%)
   - Shows compute-bound workload (good!)
   ```

### Roofline Analysis

```
Arithmetic Intensity = FLOPs / Bytes

For 1024×1024×1024 matmul:
- FLOPs: 2 × 1024³ = 2.15 billion
- Bytes: 3 × 1024² × 4 = 12.6 MB
- AI = 2147M / 12.6M = 170 FLOPs/byte

A40 Specs:
- Memory BW: 696 GB/s
- TF32 Compute: 312 TFLOPS
- Ridge point: 312T / 696G = 448 FLOPs/byte

Since 170 < 448: Still memory bound
But Tensor Cores reduce time in compute portion by 8x
```

## Limitations and Challenges

### 1. **Alignment Requirements**

Tensor Cores work best with multiples of 16:
```cuda
// Good: 1024×768×768 (all multiples of 16)
// Bad: 1000×750×750 (requires padding/masking)
```

**Our implementation issue** (kernels_req_1/matmul.cuh:45-56):
```cuda
if (bt + WMMA_M <= B * T && oc + WMMA_N <= OC && k + WMMA_K <= C) {
    // Only processes when ALL conditions met
    // Misses partial tiles at boundaries!
}
```

This means we skip computation for non-aligned dimensions, leading to:
- Incorrect results for some matrix sizes
- Wasted potential performance

**Better approach** (what cuBLAS does):
- Handle partial tiles separately
- Use predication or explicit boundary checking
- Zero-pad when necessary

### 2. **TF32 Precision**

Numerical accuracy comparison:
```
FP32 mantissa: 23 bits ≈ 7 decimal digits
TF32 mantissa: 10 bits ≈ 3 decimal digits

Relative error: ~2^-10 ≈ 0.1%
```

**For GPT-2**: Acceptable
- ML models are noise-tolerant
- Final accuracy typically within 0.1% of FP32

**Not suitable for**:
- Financial calculations
- Scientific computing requiring high precision
- Iterative solvers (error accumulation)

### 3. **Warp-Level Synchronization**

```cuda
// All 32 threads in a warp MUST execute together
wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

// Cannot have divergent warps
if (threadIdx.x < 16) {
    wmma::mma_sync(...);  // WRONG: Only half the warp
}
```

### 4. **Limited Flexibility**

Fixed tile sizes:
- 16×16×8 for TF32
- 16×16×16 for FP16
- 8×8×4 for INT8

Cannot adjust for specific workloads like register tiling allows.

## Impact on GPT-2 Forward Pass

### Computational Breakdown

For GPT-2 inference (T=50, B=1, C=768, NH=12, L=12 layers):

```
Matrix Multiplications per Layer:
1. QKV projection: (50×768) @ (768×2304) = 88M FLOPs
2. QK^T attention: 12 × (50×64) @ (64×50) = 4.6M FLOPs
3. PV attention: 12 × (50×50) @ (50×64) = 3.8M FLOPs
4. Output projection: (50×768) @ (768×768) = 59M FLOPs
5. FFN layer 1: (50×768) @ (768×3072) = 118M FLOPs
6. FFN layer 2: (50×3072) @ (3072×768) = 118M FLOPs

Total per layer: ~392M FLOPs
Total for 12 layers: ~4.7B FLOPs
```

**Matmul fraction**: ~75% of total compute

**Expected speedup**:
- Matmul: 6x faster with Tensor Cores
- Overall: 0.75 × 6 + 0.25 × 1 = 4.75x faster

**Practical speedup**: 3-5x (accounting for memory overhead, kernel launch, etc.)

## Code Walkthrough

### Critical Section: Matrix Multiplication Loop

```cuda
// kernels_req_1/matmul.cuh:76-89

for (int k = 0; k < C; k += WMMA_K) {
    if (row + WMMA_M <= T && col + WMMA_N <= HS && k + WMMA_K <= T) {
        // Load A fragment: 16×8 tile from input
        // inp is (B*T, C), we load from row 'row', column k
        wmma::load_matrix_sync(a_frag, inp + row * C + k, C);
        //                                    ^base    ^offset  ^stride

        // Load B fragment: 8×16 tile from weight (as column-major = transpose)
        // weight is (OC, C), we load from row 'col', column k
        wmma::load_matrix_sync(b_frag, weight + col * C + k, C);

        // Perform: acc_frag += a_frag @ b_frag
        // This single instruction does 2,048 FMA operations!
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
}
```

**What happens under the hood:**
1. Each thread loads part of the fragment (distributed across warp)
2. Hardware automatically converts FP32 → TF32
3. Tensor Core performs 16×16×8 matrix multiplication
4. Result accumulated in FP32 (acc_frag)
5. All 32 threads in warp participate

### Fragment Distribution

Each fragment is distributed across the 32 threads in a warp:
```
16×16 accumulator fragment:
- 256 total elements
- 256 / 32 = 8 elements per thread
- Stored in thread-local registers

Load/store handled by WMMA API:
- Automatically coalesces memory accesses
- Handles distribution across threads
- Optimizes for bank conflicts
```

## Comparison with Other Optimizations

| Feature | Baseline | req_0 | req_1 | req_2 |
|---------|----------|-------|-------|-------|
| **Hardware** | CUDA cores | CUDA cores | Tensor Cores | Tensor Cores |
| **Precision** | FP32 | FP32 | TF32→FP32 | TF32→FP32 |
| **Speedup** | 1x | 2-3x | 4-8x | 5-10x |
| **Code Complexity** | Low | Very High | Medium | Low |
| **Learning Value** | Medium | High | Very High | Low |
| **Portability** | All GPUs | All GPUs | Volta+ only | All GPUs |
| **Manual Tuning** | Some | Extensive | Some | None |

## Best Practices

### 1. **Dimension Alignment**
```cuda
// Pad matrices to multiples of 16
int padded_M = ((M + 15) / 16) * 16;
int padded_N = ((N + 15) / 16) * 16;
int padded_K = ((K + 7) / 8) * 8;
```

### 2. **Handle Partial Tiles**
```cuda
// Check boundaries for each operation
for (int k = 0; k < C; k += WMMA_K) {
    // Handle full tiles
    if (k + WMMA_K <= C) {
        wmma::load_matrix_sync(...);
    } else {
        // Handle partial tile with zero padding
        // OR use scalar code for remainder
    }
}
```

### 3. **Verify Tensor Core Usage**
```bash
# Check if Tensor Cores are actually being used
ncu --metrics sm__inst_executed_pipe_tensor.sum -k your_kernel ./program

# If zero, Tensor Cores are NOT being used!
# Common reasons:
# - Wrong fragment types
# - Misaligned dimensions
# - Divergent warp execution
```

### 4. **Mixed Precision Training**
For training (not used in this project):
```cuda
// Use FP16 for even higher throughput (640 TFLOPS on A40)
wmma::fragment<..., wmma::precision::fp16, ...> a_frag;
// But accumulate in FP32 for numerical stability
wmma::fragment<wmma::accumulator, ..., float> acc_frag;
```

## Debugging Tips

### Common Issues

1. **"Fragment dimension mismatch"**
   - Check that M, N, K match between load/mma/store operations
   - Ensure stride parameter is correct

2. **Incorrect results**
   - Verify row_major vs col_major for A and B
   - Check that all 32 threads execute wmma operations
   - Ensure proper bounds checking

3. **No speedup observed**
   - Confirm Tensor Cores are being used (check ncu metrics)
   - Verify dimensions are large enough (>256)
   - Check for memory bottlenecks

### Validation
```cuda
// Always validate against FP32 baseline
float max_error = 0.0f;
for (int i = 0; i < M * N; i++) {
    float err = fabs(output_tf32[i] - output_fp32[i]);
    max_error = max(max_error, err);
}
printf("Max error: %e\n", max_error);
// Should be < 1e-3 for TF32
```

## Files Modified

- `kernels_req_1/matmul.cuh` - Tensor Core matrix multiplication
- `kernels_req_1/attention.cuh` - Tensor Core P@V kernel
- `kernels_req_1/layernorm.cuh` - Same as baseline
- `kernels_req_1/softmax.cuh` - Same as baseline

## References

- [NVIDIA Tensor Core Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [TF32 Precision Study](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)
- NVIDIA A40 Whitepaper: Tensor Core Architecture

## Further Optimization Ideas

1. **Double buffering**: Overlap compute and memory loads
2. **Async copy**: Use `cp.async` for better memory pipelining
3. **FP16 precision**: 2x higher throughput (640 TFLOPS)
4. **Flash Attention**: Combine Tensor Cores with attention-specific optimizations
