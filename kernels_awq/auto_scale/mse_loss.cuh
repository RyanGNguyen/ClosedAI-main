#ifndef __MSE_LOSS_KERNEL_CUH__
#define __MSE_LOSS_KERNEL_CUH__

#include "../../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32
#define MSE_BLOCK_SIZE 1024

// =============================================
// MSE LOSS PARTIAL REDUCTION KERNEL (FIRST PASS)
// =============================================
// Computes partial sums of squared differences for large arrays
// Supports batched processing across L layers
// Each block computes sum for a chunk of data using grid-stride pattern
// Grid: dim3(numBlocks, L) - multiple blocks per layer
// Block: MSE_BLOCK_SIZE threads (typically 1024)
// Shared memory: numWarps * sizeof(float)
__global__ void mse_partial_kernel(
    float* partial_sums,  // Output: (L, numBlocks) - partial sum per (layer, block)
    const float* a,       // Input array A: (L, elements_per_layer)
    const float* b,       // Input array B: (L, elements_per_layer)
    int L,                // Number of layers
    int elements_per_layer // Number of elements per layer
) {
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warpId = tid / WARP_SIZE;
    int block_threads = blockDim.x;
    int numWarps = CEIL_DIV(block_threads, WARP_SIZE);
    int l = blockIdx.y;   // Layer index
    
    // Shared memory for warp-level results
    extern __shared__ float shmem[];
    float* sh_sum = shmem;  // [0 .. numWarps-1]
    
    // Step 1: Each thread accumulates local sum using grid-stride pattern
    float local_sum = 0.0f;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;
    
    if (l < L) {
        // Offset to this layer's data
        const float* a_layer = a + l * elements_per_layer;
        const float* b_layer = b + l * elements_per_layer;
        
        for (int i = global_idx; i < elements_per_layer; i += grid_stride) {
            float diff = a_layer[i] - b_layer[i];
            local_sum += diff * diff;
        }
    }
    
    // Step 2: Warp-level reduction using shuffle operations
    unsigned mask = 0xffffffffu;
    for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float peer_sum = __shfl_down_sync(mask, local_sum, offset);
        local_sum += peer_sum;
    }
    
    // Step 3: Store one result per warp to shared memory
    if (lane == 0) {
        sh_sum[warpId] = local_sum;
    }
    __syncthreads();
    
    // Step 4: Final block-wide reduction (first warp only)
    if (warpId == 0) {
        float warp_sum = (lane < numWarps) ? sh_sum[lane] : 0.0f;
        
        // Warp-level reduction
        for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            float peer_sum = __shfl_down_sync(mask, warp_sum, offset);
            warp_sum += peer_sum;
        }
        
        // Step 5: Thread 0 writes partial sum for this (layer, block)
        if (lane == 0 && l < L) {
            partial_sums[l * gridDim.x + blockIdx.x] = warp_sum;
        }
    }
}

// =============================================
// MSE LOSS FINAL REDUCTION KERNEL (SECOND PASS)
// =============================================
// Reduces partial sums from all blocks to final MSE value per layer
// Supports batched processing across L layers
// Grid: L blocks (one per layer)
// Block: MSE_BLOCK_SIZE threads (typically 1024)
// Shared memory: numWarps * sizeof(float)
__global__ void mse_final_kernel(
    float* mse_output,         // Output: (L,) - one MSE value per layer
    const float* partial_sums, // Input: (L, num_partials) - partial sum per (layer, block)
    int L,                     // Number of layers
    int num_partials,          // Number of partial sums per layer
    int elements_per_layer     // Number of elements per layer (for averaging)
) {
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warpId = tid / WARP_SIZE;
    int block_threads = blockDim.x;
    int numWarps = CEIL_DIV(block_threads, WARP_SIZE);
    int l = blockIdx.x;        // Layer index (one block per layer)
    
    // Shared memory for warp-level results
    extern __shared__ float shmem[];
    float* sh_sum = shmem;  // [0 .. numWarps-1]
    
    // Step 1: Each thread accumulates partial sums for this layer
    float local_sum = 0.0f;
    if (l < L) {
        const float* layer_partials = partial_sums + l * num_partials;
        for (int i = tid; i < num_partials; i += block_threads) {
            local_sum += layer_partials[i];
        }
    }
    
    // Step 2: Warp-level reduction using shuffle operations
    unsigned mask = 0xffffffffu;
    for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float peer_sum = __shfl_down_sync(mask, local_sum, offset);
        local_sum += peer_sum;
    }
    
    // Step 3: Store one result per warp to shared memory
    if (lane == 0) {
        sh_sum[warpId] = local_sum;
    }
    __syncthreads();
    
    // Step 4: Final block-wide reduction (first warp only)
    if (warpId == 0) {
        float warp_sum = (lane < numWarps) ? sh_sum[lane] : 0.0f;
        
        // Warp-level reduction
        for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            float peer_sum = __shfl_down_sync(mask, warp_sum, offset);
            warp_sum += peer_sum;
        }
        
        // Step 5: Thread 0 computes MSE and writes final result for this layer
        if (lane == 0 && l < L) {
            mse_output[l] = warp_sum / (float)elements_per_layer;
        }
    }
}

// Batched MSE loss (all layers in parallel)
// Computes Mean Squared Error for L layers in parallel
// Results stay on device for efficient accumulation in grid search
void mse_loss(
    float* d_mse_per_layer,    // Output: (L,) - MSE per layer (device pointer)
    const float* d_a,           // Input: (L, elements_per_layer)
    const float* d_b,           // Input: (L, elements_per_layer)
    int L,                      // Number of layers
    int elements_per_layer      // B * T * OC
) {
    int blockSize = MSE_BLOCK_SIZE;
    int numWarps = blockSize / WARP_SIZE;
    size_t shmem_size = numWarps * sizeof(float);
    
    // Calculate number of blocks for first pass
    int numBlocks = min(CEIL_DIV(elements_per_layer, blockSize), 1024);  // Cap at 1024 blocks
    
    // Allocate device memory for partial sums: (L, numBlocks)
    float *d_partial_sums;
    cudaCheck(cudaMalloc(&d_partial_sums, L * numBlocks * sizeof(float)));
    
    // First pass: compute partial sums across multiple blocks for all layers
    dim3 grid1(numBlocks, L);  // L layers processed in parallel
    mse_partial_kernel<<<grid1, blockSize, shmem_size>>>(
        d_partial_sums, d_a, d_b, L, elements_per_layer);
    cudaCheck(cudaGetLastError());
    
    // Second pass: reduce partial sums to final MSE value per layer
    // One block per layer
    mse_final_kernel<<<L, blockSize, shmem_size>>>(
        d_mse_per_layer, d_partial_sums, L, numBlocks, elements_per_layer);
    cudaCheck(cudaGetLastError());
    
    // Cleanup
    cudaCheck(cudaFree(d_partial_sums));
}

#endif // __MSE_LOSS_KERNEL_CUH__
