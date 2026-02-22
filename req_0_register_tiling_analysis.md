# req_0: Register Tiling Optimization

## Overview

Register tiling (also known as thread tiling) is an advanced GPU optimization technique that combines shared memory tiling with register-level tiling to maximize data reuse and arithmetic intensity. This optimization was applied to matrix multiplication kernels in the GPT-2 forward pass.

## What Is Register Tiling?

Register tiling adds an additional level of blocking on top of shared memory tiling:

```
Global Memory → Shared Memory (Block-level tiling) → Registers (Thread-level tiling) → Compute
```

### Memory Hierarchy:
- **Global Memory**: ~500 GB/s bandwidth, ~200 cycle latency
- **Shared Memory**: ~10 TB/s bandwidth, ~20 cycle latency
- **Registers**: ~100+ TB/s bandwidth, 1-2 cycle latency

By storing frequently accessed data in registers, we minimize even shared memory accesses.

## Implementation Details

### Baseline vs Optimized Comparison

**Baseline** (`kernels/matmul.cuh`):
```cuda
#define TILE_WIDTH 16
// Each thread computes 1 output element
// Shared memory tiles: 16x16
// Thread block: 16x16 = 256 threads
```

**Optimized** (`kernels_req_0/matmul.cuh`):
```cuda
#define TILE_M 64    // Shared memory tile height
#define TILE_N 64    // Shared memory tile width
#define TILE_S 8     // Shared memory tile depth (K dimension)

#define TM 8         // Register tile height per thread
#define TN 8         // Register tile width per thread

#define BM (TILE_M/TM)  // = 8 threads
#define BN (TILE_N/TN)  // = 8 threads
// Thread block: 8x8 = 64 threads
```

### Key Architectural Changes

#### 1. **Larger Shared Memory Tiles**
- Baseline: 16x16 tiles
- Optimized: 64x64 output tile, with 64x8 and 8x64 input tiles

#### 2. **Register Arrays per Thread**
Each thread maintains an 8x8 register tile:
```cuda
float acc[TM][TN];  // 64 registers per thread for accumulation
float a_reg[TM];    // 8 registers for A matrix values
float b_reg[TN];    // 8 registers for B matrix values
```

#### 3. **Computation Pattern**
```cuda
// Load 8x8 tile from shared memory to registers
for(int i = 0; i < TM; i++) {
    a_reg[i] = As[ty * TM + i][k];
}
for(int j = 0; j < TN; j++) {
    b_reg[j] = Bs[k][tx * TN + j];
}

// Compute 8x8 outer product in registers
for(int i = 0; i < TM; i++) {
    for(int j = 0; j < TN; j++) {
        acc[i][j] += a_reg[i] * b_reg[j];  // 64 FMA ops
    }
}
```

### Applied Kernels

#### 1. **Matrix Multiplication** (kernels_req_0/matmul.cuh)
- Main optimization target
- Two kernel variants:
  - `matmul_forward_kernel`: Optimized for large matrices (≥64x64)
  - `matmul_forward_kernel_small`: Fallback for small matrices

**Location**: kernels_req_0/matmul.cuh:20-133

#### 2. **Attention P@V Multiplication** (kernels_req_0/attention.cuh)
- Custom kernel: `pv_matmul_optimized_kernel`
- Tiles: 32x32 output with 4x4 register tiling
- Applied to the P @ V operation in attention mechanism

**Location**: kernels_req_0/attention.cuh:51-147

## Why This Optimization Is Effective

### 1. **Reduced Memory Traffic**
- **Baseline**: Each value loaded from shared memory once per use
- **Optimized**: Each value loaded from shared memory once, then reused 8 times from registers

**Example**: For one K-tile iteration
- Baseline: 8 shared memory loads → 8 FMA operations (1:1 ratio)
- Optimized: 16 shared memory loads → 64 FMA operations (1:4 ratio)

### 2. **Higher Arithmetic Intensity**
```
Arithmetic Intensity = FLOPs / Bytes Loaded

Baseline (16x16 tile, 1x1 register):
- Thread loads: 2 float/iter = 8 bytes
- Thread computes: 2 FLOPs
- AI = 2/8 = 0.25 FLOPs/byte

Optimized (64x64 tile, 8x8 register):
- Thread loads: 16 float/iter = 64 bytes
- Thread computes: 128 FLOPs
- AI = 128/64 = 2 FLOPs/byte

Improvement: 8x higher arithmetic intensity
```

### 3. **Better Instruction-Level Parallelism (ILP)**
- 64 independent accumulator registers allow GPU to schedule multiple instructions in parallel
- Reduces pipeline stalls and improves throughput
- Modern GPUs can execute 4+ independent instructions per cycle

### 4. **Improved Occupancy**
- Baseline: 256 threads/block × many blocks = high register pressure
- Optimized: 64 threads/block allows more blocks per SM
- Better balance between parallelism and resource usage

## Performance Analysis

### Expected Speedup
| Matrix Size | Expected Speedup vs Baseline |
|-------------|------------------------------|
| 64 × 64     | 1.5-2x                       |
| 256 × 256   | 2-2.5x                       |
| 1024 × 1024 | 2.5-3x                       |
| 2048 × 2048 | 2.5-3x                       |

### Key Metrics to Profile

Using `ncu` (Nsight Compute):

1. **Memory Throughput**
   - Look for: Lower shared memory bandwidth utilization
   - Expected: 30-50% reduction in shared memory traffic

2. **SM Efficiency**
   - Look for: Higher SM efficiency (>80%)
   - Shows better utilization of compute units

3. **Achieved Occupancy**
   - Look for: 50-75% occupancy
   - Lower thread count but better resource balance

4. **Pipeline Utilization**
   - Look for: Higher ILP score
   - More instructions in flight simultaneously

### Profiling Commands
```bash
# Kernel-level profiling
ncu --set full -o matmul_req0 ./next_token_generation

# Focus on specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              dram__throughput.avg.pct_of_peak_sustained_elapsed,\
              l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    -k matmul_forward_kernel ./next_token_generation
```

## Limitations and Trade-offs

### 1. **Register Pressure**
- Each thread uses ~80 registers (64 for acc + 16 for a_reg/b_reg)
- May limit occupancy on some GPUs
- Trade-off: Fewer concurrent threads but more work per thread

### 2. **Complex Code**
- More difficult to implement and debug
- Many #pragma unroll directives increase code size
- Harder to maintain

### 3. **Fixed Tile Sizes**
- Optimal tile sizes depend on matrix dimensions
- Current implementation (64x64) may not be optimal for all cases
- Small matrices use fallback kernel

### 4. **Edge Cases**
- Non-multiple-of-64 dimensions require careful boundary checking
- Padding or boundary handling adds overhead

## Impact on GPT-2 Forward Pass

### Where This Matters Most

GPT-2 has many matrix multiplications:
1. **Attention QK^T**: (B×NH×T×HS) @ (B×NH×HS×T) = (B×NH×T×T)
2. **Attention PV**: (B×NH×T×T) @ (B×NH×T×HS) = (B×NH×T×HS)
3. **FFN Layer 1**: (B×T×C) @ (C×4C) = (B×T×4C)
4. **FFN Layer 2**: (B×T×4C) @ (4C×C) = (B×T×C)

For typical dimensions (B=1, T=50, C=768, NH=12):
- Attention matmuls: Small-to-medium (64×64 to 1024×64)
- FFN matmuls: Medium-to-large (50×768 @ 768×3072)

**Expected overall speedup**: 1.5-2x compared to baseline

## Code Walkthrough

### Critical Section: Inner Loop
```cuda
// kernels_req_0/matmul.cuh:92-116
for(int k = 0; k < TILE_S; k++) {
    // Step 1: Load from shared memory to registers
    float a_reg[TM];  // 8 values
    float b_reg[TN];  // 8 values

    #pragma unroll
    for(int i = 0; i < TM; i++) {
        a_reg[i] = As[ty * TM + i][k];  // Coalesced read
    }

    #pragma unroll
    for(int j = 0; j < TN; j++) {
        b_reg[j] = Bs[k][tx * TN + j];  // Coalesced read
    }

    // Step 2: Compute 8×8 outer product (64 FMA operations)
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            acc[i][j] += a_reg[i] * b_reg[j];
        }
    }
}
```

**Why this works:**
- 16 shared memory reads → 64 FMA operations
- All 64 FMA operations use register operands (1-2 cycle latency)
- High instruction throughput due to independent accumulators

## Comparison with Other Optimizations

| Aspect | Baseline | req_0 (Register) | req_1 (Tensor) | req_2 (cuBLAS) |
|--------|----------|------------------|----------------|----------------|
| Complexity | Low | High | Medium | Low |
| Speedup | 1x | 2-3x | 4-8x | 5-10x |
| Portability | High | High | Medium (needs Volta+) | High |
| Learning Value | Medium | High | Very High | Low |

## Recommendations

1. **Use for learning**: Excellent for understanding GPU memory hierarchy
2. **Production use**: Consider cuBLAS instead for matmul, but use register tiling for custom kernels
3. **Further optimization**: Combine with double buffering and prefetching for even better performance

## References

- ECE 508 Lecture 4: Advanced Tiling Techniques
- NVIDIA CUDA Best Practices Guide: Register Pressure
- "MAGMA: Matrix Algebra on GPU and Multicore Architectures" - Uses similar techniques

## Files Modified

- `kernels_req_0/matmul.cuh` - Main matrix multiplication kernel
- `kernels_req_0/attention.cuh` - Custom P@V kernel with register tiling
- `kernels_req_0/layernorm.cuh` - Same as baseline (reduction used instead)
- `kernels_req_0/softmax.cuh` - Same as baseline (reduction used instead)