#ifndef __NORMALIZE_SCALES_KERNEL_CUH__
#define __NORMALIZE_SCALES_KERNEL_CUH__

#include "../../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32

// =============================================
// ELEMENT-WISE POWER KERNEL
// =============================================
// Applies __powf(base, exponent) to each element
// Grid: ceil(size / 1024) blocks
// Block: 1024 threads
__global__ void pow_kernel(float* data, int size, float exponent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = __powf(data[idx], exponent);
    }
}

// =============================================
// GLOBAL MIN/MAX REDUCTION KERNEL
// =============================================
// Single-block reduction to find global min and max
// Grid: 1 block
// Block: 1024 threads (32 warps)
// Shared memory: 2 * 32 * sizeof(float) = 256 bytes
__global__ void minmax_kernel(
    float* global_min,    // Output: single min value
    float* global_max,    // Output: single max value
    const float* data,    // Input: (L, C) array
    int size              // L * C
) {
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warpId = tid / WARP_SIZE;
    int block_threads = blockDim.x;
    int numWarps = CEIL_DIV(block_threads, WARP_SIZE);
    
    // Shared memory for warp-level results
    extern __shared__ float shmem[];
    float* sh_min = shmem;              // [0 .. numWarps-1]
    float* sh_max = shmem + numWarps;   // [numWarps .. 2*numWarps-1]
    
    // Step 1: Each thread accumulates local min/max over strided elements
    float local_min = FLT_MAX;
    float local_max = 0.0f;
    
    for (int i = tid; i < size; i += block_threads) {
        float val = data[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }
    
    // Step 2: Warp-level reduction using shuffle operations
    unsigned mask = 0xffffffffu;
    for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float min_peer = __shfl_down_sync(mask, local_min, offset);
        float max_peer = __shfl_down_sync(mask, local_max, offset);
        local_min = fminf(local_min, min_peer);
        local_max = fmaxf(local_max, max_peer);
    }
    
    // Step 3: Store one result per warp to shared memory
    if (lane == 0) {
        sh_min[warpId] = local_min;
        sh_max[warpId] = local_max;
    }
    __syncthreads();
    
    // Step 4: Final block-wide reduction (first warp only)
    if (warpId == 0) {
        // Scales already are in absolute magnitude
        float warp_min = (lane < numWarps) ? sh_min[lane] : FLT_MAX;
        float warp_max = (lane < numWarps) ? sh_max[lane] : 0.0f;
        
        // Warp-level reduction
        for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            float min_peer = __shfl_down_sync(mask, warp_min, offset);
            float max_peer = __shfl_down_sync(mask, warp_max, offset);
            warp_min = fminf(warp_min, min_peer);
            warp_max = fmaxf(warp_max, max_peer);
        }
        
        // Step 5: Thread 0 writes final result
        if (lane == 0) {
            *global_min = warp_min;
            *global_max = warp_max;
        }
    }
}

// =============================================
// ELEMENT-WISE DIVISION KERNEL (SHARED UTILITY)
// =============================================
// Divides each element by a scalar divisor
// Grid: ceil(size / blockSize) blocks
// Block: configurable (typically 1024 or 768 threads)
// Used by: normalize_scales, saliency.cuh (finalize_averages)
__global__ void elementwise_division_kernel(float* data, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= divisor;
    }
}

// =============================================
// HOST WRAPPER FUNCTION
// =============================================
// Normalizes an L x C array through 3 steps:
// 1. Exponentiate: data[i] = pow(data[i], exponent)
// 2. Find global min and max
// 3. Normalize: data[i] /= sqrt(max * min)
void normalize_scales(
    float* d_data,     // In/out: (L, C) array on device
    int L,             // Number of layers (12)
    int C,             // Number of channels (768)
    float exponent     // Arbitrary exponent for power operation
) {
    int size = L * C;
    int blockSize = 1024;
    int numBlocks = CEIL_DIV(size, blockSize); // 9
    
    // Step 1: Exponentiate all elements
    pow_kernel<<<numBlocks, blockSize>>>(d_data, size, exponent);
    cudaCheck(cudaGetLastError());
    
    // Step 2: Find global min/max using single-block reduction
    float *d_min, *d_max;
    cudaCheck(cudaMalloc(&d_min, sizeof(float)));
    cudaCheck(cudaMalloc(&d_max, sizeof(float)));
    
    int numWarps = blockSize / WARP_SIZE;  // 32 warps
    size_t shmem_size = 2 * numWarps * sizeof(float);  // 256 bytes
    
    minmax_kernel<<<1, blockSize, shmem_size>>>(d_min, d_max, d_data, size);
    cudaCheck(cudaGetLastError());
    
    // Step 3: Compute scale factor on host: sqrt(max * min)
    float h_min, h_max;
    cudaCheck(cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    
    float scale = sqrtf(h_max * h_min);
    
    // Step 4: Normalize all elements by the scale factor
    elementwise_division_kernel<<<numBlocks, blockSize>>>(d_data, size, scale);
    cudaCheck(cudaGetLastError());
    
    // Cleanup
    cudaCheck(cudaFree(d_min));
    cudaCheck(cudaFree(d_max));
}

#endif // __NORMALIZE_SCALES_KERNEL_CUH__
