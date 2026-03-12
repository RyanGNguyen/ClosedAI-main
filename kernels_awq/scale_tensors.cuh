#ifndef __SCALE_TENSORS_KERNEL_CUH__
#define __SCALE_TENSORS_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>

// =============================================
// SHARED OPERATION FUNCTORS
// =============================================

// Operation functors for generic channel-wise scaling
struct MultiplyOp {
    __device__ __forceinline__ float operator()(float value, float scale) const {
        return value * scale;
    }
};

struct DivideOp {
    __device__ __forceinline__ float operator()(float value, float scale) const {
        return value / scale;
    }
};

// =============================================
// UNIFIED CHANNEL SCALING KERNEL
// =============================================

// Generic kernel to apply per-layer, per-channel scale factors to any (L, N, C) tensor
// This unified kernel works for both activations and weights.
//
// For activations: N = B*T (batch size × sequence length)
// For weights:     N = OC (number of output channels)
//
// Input data: (L, N, C) - L layers, N elements along middle dimension, C channels
// Input scales: (L, C) - per-layer scale factors for each channel
// Operation: data[l, n, c] = Op(data[l, n, c], scales[l, c])
//
// Memory layout (row-major):
//   data[l][n][c] -> data[(l * N + n) * C + c]
//   scales[l][c] -> scales[l * C + c]
//
// Each thread handles one (l, n, c) element. Threads are organized to
// maximize memory coalescing: adjacent threads access adjacent channels.
//
// Grid: dim3(CEIL_DIV(C, 1024), N, L)
// Block: 1024 threads
template<typename Op>
__global__ void apply_channel_scales_kernel(
    float* d_out,               // Output: (L, N, C) - can be in-place
    const float* d_in,          // Input: (L, N, C) - can be same as output
    const float* d_scales,      // Scales: (L, C) - per-layer channel scales
    int L, int N, int C,
    Op op)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // Channel index (0 to C-1)
    int n = blockIdx.y;                              // N dimension index (0 to N-1)
    int l = blockIdx.z;                              // Layer index (0 to L-1)
    
    if (c < C) {
        // Load scale for this (layer, channel) pair
        float scale = d_scales[l * C + c];
        
        // Calculate data index: (l * N + n) * C + c
        int idx = (l * N + n) * C + c;
        
        // Apply operation: multiply, divide, etc.
        d_out[idx] = op(d_in[idx], scale);
    }
}

// Scale channels (division: X_scaled = X * s)
// Supports both in-place (d_out == d_in) and out-of-place scaling
//
// Usage examples:
//   - Activations: scale_channels(d_act, d_act, d_scales, L, B*T, C)
//   - Weights:     scale_channels(d_weights, d_weights, d_scales, L, OC, C)
void scale_channels(
    float* d_out,               // Output: (L, N, C)
    const float* d_in,          // Input: (L, N, C) - can be same as output
    const float* d_scales,      // Scales: (L, C)
    int L, int N, int C)
{
    // Grid: (ceil(C/1024), N, L)
    // Block: 1024 threads - one thread per channel
    dim3 grid(CEIL_DIV(C, 1024), N, L);
    dim3 block(1024);
    
    apply_channel_scales_kernel<<<grid, block>>>(
        d_out, d_in, d_scales, L, N, C, MultiplyOp());
    cudaCheck(cudaGetLastError());
}

// Unscale channels (multiplication: X = X / s)
// Supports both in-place (d_out == d_in) and out-of-place unscaling
//
// Usage examples:
//   - Activations: unscale_channels(d_act, d_act, d_scales, L, B*T, C)
//   - Weights:     unscale_channels(d_weights, d_weights, d_scales, L, OC, C)
void unscale_channels(
    float* d_out,               // Output: (L, N, C)
    const float* d_in,          // Input: (L, N, C) - can be same as output
    const float* d_scales,      // Scales: (L, C)
    int L, int N, int C)
{
    // Grid: (ceil(C/1024), N, L)
    // Block: 1024 threads - one thread per channel
    dim3 grid(CEIL_DIV(C, 1024), N, L);
    dim3 block(1024);
    
    apply_channel_scales_kernel<<<grid, block>>>(
        d_out, d_in, d_scales, L, N, C, DivideOp());
    cudaCheck(cudaGetLastError());
}

// =============================================
// AWQ WEIGHT OUTPUT ROW SCALING
// =============================================

// Kernel to scale FC weight output rows (middle dimension) by dividing with scales
// Used in scale_fc_fc pattern: fc.weight.div_(scales.view(-1, 1))
// Supports row_offset for partial scaling (e.g., V weights in qkvw)
//
// Grid: dim3(CEIL_DIV(IC, 1024), num_rows, L)
// Block: 1024 threads
__global__ void scale_fc_output_rows_kernel(
    float* d_weights,            // In/Out: (L, OC, IC) - FC weights
    const float* d_scales,       // Input: (L, num_rows) - scale factors for output rows
    int L, int OC, int IC,
    int row_offset,              // Starting row to scale
    int num_rows)                // Number of rows to scale
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x;  // Input channel index
    int row = blockIdx.y;                             // Row index (0 to num_rows-1)
    int l = blockIdx.z;                               // Layer index
    
    if (ic < IC && row < num_rows && l < L) {
        int actual_row = row_offset + row;  // Actual row in weight matrix
        int weight_idx = (l * OC + actual_row) * IC + ic;
        int scale_idx = l * num_rows + row;
        
        d_weights[weight_idx] /= d_scales[scale_idx];
    }
}

// Host launcher: scale FC weight output rows (in-place division)
void scale_fc_output_rows(
    float* d_weights,            // In/Out: (L, OC, IC)
    const float* d_scales,       // Input: (L, num_rows)
    int L, int OC, int IC,
    int row_offset,              // Starting row to scale (0 for full, 2*C for V weights)
    int num_rows)                // Number of rows to scale
{
    dim3 grid(CEIL_DIV(IC, 1024), num_rows, L);
    dim3 block(1024);
    scale_fc_output_rows_kernel<<<grid, block>>>(
        d_weights, d_scales, L, OC, IC, row_offset, num_rows);
    cudaCheck(cudaGetLastError());
}

// =============================================
// SALIENT CHANNEL SCALING
// =============================================

// Kernel to build scale tensor from salient indices in a single pass
// Salient channels get 2.0, all others get 1.0
// Grid: dim3(CEIL_DIV(C, 1024), L)
// Block: 1024 threads
__global__ void build_salient_scales_kernel(
    float* d_scales,              // Output: (L, C)
    const uint32_t* d_indices,    // Input: (L * k) salient channel indices
    int L, int C, int k)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // Channel index
    int l = blockIdx.y;                              // Layer index
    
    if (c < C && l < L) {
        // Default to 1.0
        float scale = 1.0f;
        
        // Check if this channel is in the salient list
        const uint32_t* layer_indices = d_indices + l * k;
        for (int i = 0; i < k; i++) {
            if (layer_indices[i] == (uint32_t)c) {
                scale = 2.0f;
                break;
            }
        }
        
        d_scales[l * C + c] = scale;
    }
}

// Host launcher: builds scale tensor from salient indices
// Salient channels get 2.0x scaling, all others get 1.0x
void build_salient_scales(
    float* d_scales,              // Output: (L, C)
    const uint32_t* d_indices,    // Input: (L * k) salient channel indices on device
    int L, int C, int k)
{
    dim3 grid(CEIL_DIV(C, 1024), L);
    dim3 block(1024);
    build_salient_scales_kernel<<<grid, block>>>(d_scales, d_indices, L, C, k);
    cudaCheck(cudaGetLastError());
}

#endif // __SCALE_TENSORS_KERNEL_CUH__
