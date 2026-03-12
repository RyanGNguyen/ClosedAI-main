#ifndef __SALIENCY_KERNEL_CUH__
#define __SALIENCY_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_DIM_X 32   // Channels per block (warp width for coalescing)
#define BLOCK_DIM_Y 32   // Threads for N dimension reduction
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)  // = 1024 threads per block


// =============================================
// REDUCTION KERNELS FOR CHANNEL-WISE OPERATIONS
// =============================================

// Finds sum of absolute values along N dimension for each (layer, channel) pair
// Input (activation): (L, N, C) where N = B*T
// Output (partial_sums): (NUM_N_BLOCKS, L, C) where NUM_N_BLOCKS = ceil(N / BLOCK_DIM_Y)
// Grid:  3D grid (ceil(C / BLOCK_DIM_X), NUM_N_BLOCKS, L)
// Block: dim3(BLOCK_DIM_X, BLOCK_DIM_Y) = (32, 32) = 1024 threads
// Uses grid-stride pattern in Y dimension for coalesced access
__global__ void channel_asum_kernel(
    float *partial_sums,  // (NUM_N_BLOCKS, L, C)
    const float *act,     // (L, N, C)
    int L, int N, int C
) {
    int layer = blockIdx.z;  // Layer index (0 to L-1)
    
    // 2D thread indexing for clean memory access
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // Global channel index
    int n_start = blockIdx.y * blockDim.y + threadIdx.y;  // Starting row for this thread
    int n_stride = gridDim.y * blockDim.y;  // Grid-stride over N
    
    // Each thread accumulates sum for ONE (layer, channel) pair across subset of N
    float localSum = 0.0f;
    
    if (c < C) {
        // Grid-stride loop over N dimension
        for (int n = n_start; n < N; n += n_stride) {
            float val = fabsf(act[(layer * N + n) * C + c]);
            localSum += val;
        }
    }
    
    // Shared memory: partials[row][col]
    __shared__ float partials[BLOCK_DIM_Y][BLOCK_DIM_X];
    partials[threadIdx.y][threadIdx.x] = localSum;
    __syncthreads();
    
    // Logarithmic reduction over rows (Y dimension)
    #pragma unroll
    for (int stride = BLOCK_DIM_Y >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            partials[threadIdx.y][threadIdx.x] += partials[threadIdx.y + stride][threadIdx.x];
        }
        __syncthreads();
    }
    
    // Row 0 threads write the final reduced value
    if (threadIdx.y == 0 && c < C) {
        partial_sums[(blockIdx.y * L + layer) * C + c] = partials[0][threadIdx.x];
    }
}

// Second pass: Reduce partial sums across N blocks
// Input (partial_sums): (NUM_N_BLOCKS, L, C)
// Output (output): (L, C) - accumulates to existing values
// Grid: (ceil(C / 256), L)
// Block: 256 threads
__global__ void reduce_partials_kernel(
    float *output,             // (L, C) - accumulator
    const float *partial_sums, // (NUM_N_BLOCKS, L, C)
    int NUM_N_BLOCKS, int L, int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = blockIdx.y;
    
    if (c < C) {
        float sum = 0.0f;
        // Sum over all partial blocks
        for (int b = 0; b < NUM_N_BLOCKS; b++) {
            sum += partial_sums[(b * L + layer) * C + c];
        }
        // Accumulate to output (for running calibration average)
        output[layer * C + c] += sum;
    }
}

// Batched calibration: Accumulate channel sums for entire (L, N, C) tensor
// Processes all layers at once, significantly reducing kernel launch overhead
void accumulate_salient_channels(
    float* d_channel_sum_global,  // Accumulator: (L, C)
    const float* d_in,             // Activations: (L, N, C) where N = B*T
    int L, int N, int C
) {
    // Calculate number of blocks needed in Y dimension
    int NUM_N_BLOCKS = CEIL_DIV(N, BLOCK_DIM_Y);
    
    // Allocate intermediate buffer for partial sums: (NUM_N_BLOCKS, L, C)
    float *partial_sums;
    size_t partial_size = NUM_N_BLOCKS * L * C * sizeof(float);
    cudaCheck(cudaMalloc(&partial_sums, partial_size));
    
    // First pass: reduce (L, N, C) -> (NUM_N_BLOCKS, L, C)
    dim3 grid1(CEIL_DIV(C, BLOCK_DIM_X), NUM_N_BLOCKS, L);
    dim3 block1(BLOCK_DIM_X, BLOCK_DIM_Y);
    channel_asum_kernel<<<grid1, block1>>>(partial_sums, d_in, L, N, C);
    
    // Second pass: reduce (NUM_N_BLOCKS, L, C) -> (L, C)
    dim3 grid2(CEIL_DIV(C, 256), L);
    dim3 block2(256);
    reduce_partials_kernel<<<grid2, block2>>>(d_channel_sum_global, partial_sums, NUM_N_BLOCKS, L, C);
    
    cudaCheck(cudaFree(partial_sums));
}

// ============================================================================
// FINALIZATION KERNELS (used after calibration)
// ============================================================================

// Divides each element by a scalar divisor
// Grid: ceil(size / blockSize) blocks
// Block: configurable (typically 1024 or 768 threads)
// Used by: normalize_scales, saliency.cuh (finalize_averages)
__global__ void elementwise_divide_kernel(float* data, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= divisor;
    }
}

// Convert accumulated sums to averages by dividing by total token count
void finalize_averages(float* d_channel_data, int total_channels, size_t total_tokens) {
    int threads = 768; // Align with C
    int blocks = CEIL_DIV(total_channels, threads);
    
    // Reuse the element-wise division kernel from common utilities
    extern __global__ void elementwise_divide_kernel(float* data, int size, float divisor);
    elementwise_divide_kernel<<<blocks, threads>>>(
        d_channel_data, total_channels, (float)total_tokens
    );
    cudaCheck(cudaDeviceSynchronize());
}

#endif // __SALIENCY_KERNEL_CUH__
