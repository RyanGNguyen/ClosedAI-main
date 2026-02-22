#ifndef GELU_KERNEL_CUH_
#define GELU_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include <math_constants.h>

// ============================================================================
// CONSTANT MEMORY OPTIMIZATION (req_6)
// ============================================================================
// Store GELU mathematical constants in constant memory
// Benefits:
// 1. These constants are read by every thread in every GELU invocation
// 2. Perfect uniform access pattern - all threads read the same value
// 3. Cached in constant cache with broadcast capability
// 4. Small size (8 bytes total) - well within 64KB constant memory limit
// 5. Eliminates repeated register initialization
// ============================================================================

__constant__ float c_gelu_coeff;     // 0.044715f
__constant__ float c_gelu_scale;     // sqrt(2/pi)

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = inp[idx];
        // CONSTANT MEMORY OPTIMIZATION: Using c_gelu_coeff and c_gelu_scale
        // from constant memory instead of immediate constants
        float cube = c_gelu_coeff * val * val * val;
        out[idx] = 0.5f * val * (1.0f + tanhf(c_gelu_scale * (val + cube)));
    }
}

// Launch kernel here
void gelu_forward(float* out, const float* inp, int N) {
    // ============================================================================
    // CONSTANT MEMORY SETUP (done once at initialization)
    // Initialize GELU constants in constant memory
    // This is a one-time cost that benefits all subsequent kernel invocations
    // ============================================================================
    static bool constants_initialized = false;
    if (!constants_initialized) {
        float gelu_coeff = 0.044715f;
        float gelu_scale = sqrtf(2.0f / CUDART_PI_F);
        cudaMemcpyToSymbol(c_gelu_coeff, &gelu_coeff, sizeof(float));
        cudaMemcpyToSymbol(c_gelu_scale, &gelu_scale, sizeof(float));
        constants_initialized = true;
    }

    unsigned blockSize = 32;
    dim3 blockDims(blockSize);
    dim3 gridDims(CEIL_DIV(N, blockSize));

    gelu_forward_kernel<<<gridDims, blockDims>>>(out, inp, N);
}

#endif // GELU_KERNEL_CUH_
