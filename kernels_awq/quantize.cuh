#ifndef __QUANTIZE_KERNEL_CUH__
#define __QUANTIZE_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#ifndef QUANT_BLOCK_SIZE
#define QUANT_BLOCK_SIZE 1024
#endif
#define GROUP_SIZE 128
#define OC_PER_BLOCK (QUANT_BLOCK_SIZE / GROUP_SIZE)  // = 8


// ============================================================================
// GROUP QUANTIZATION PARAMETER COMPUTATION KERNEL
// ============================================================================

// First, performs amax and amin reduction per group
// Then, calculates each group's quantization factor and zero point
// IC = input channels, OC = output channels, num_groups = IC / GROUP_SIZE
// Input (weights): (L, OC, IC) - supports batched processing across layers
// Output (q_factor, zero_point): (L, OC, num_groups)
__global__ void calc_quant_param_kernel(
    float *q_factor, uint8_t *zero_point, const float *weight, int L, int OC, int IC
) {
    // Shared memory: partials[row][col] where row=OC, col=IC_element
    __shared__ float max_partials[OC_PER_BLOCK][GROUP_SIZE];
    __shared__ float min_partials[OC_PER_BLOCK][GROUP_SIZE];

    // 3D thread indexing for batched processing
    int ic = blockIdx.x * blockDim.x + threadIdx.x;   // Global input channel index (cols)
    int oc = blockIdx.y * blockDim.y + threadIdx.y;   // Global output channel index (rows)
    int l = blockIdx.z;                                // Layer index
    
    // Each thread reads exactly ONE element from this layer
    if (ic < IC && oc < OC && l < L) {
        int weight_idx = (l * OC + oc) * IC + ic;
        float val = fabsf(weight[weight_idx]);
        max_partials[threadIdx.y][threadIdx.x] = val;
        min_partials[threadIdx.y][threadIdx.x] = val;
    } else {
        max_partials[threadIdx.y][threadIdx.x] = 0.0f;
        min_partials[threadIdx.y][threadIdx.x] = FLT_MAX;
    }
    __syncthreads();
    
    // Logarithmic reduction over columns (X dimension) - reduce GROUP_SIZE elements
    #pragma unroll
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            max_partials[threadIdx.y][threadIdx.x] = fmaxf(
                max_partials[threadIdx.y][threadIdx.x],
                max_partials[threadIdx.y][threadIdx.x + stride]
            );
            min_partials[threadIdx.y][threadIdx.x] = fminf(
                min_partials[threadIdx.y][threadIdx.x],
                min_partials[threadIdx.y][threadIdx.x + stride]
            );
        }
        __syncthreads();
    }
    
    // Column 0 threads compute each group's quantization parameters
    if (threadIdx.x == 0 && oc < OC && l < L) {
        // Load group min/max
        float group_max = max_partials[threadIdx.y][0];
        float group_min = min_partials[threadIdx.y][0];
        
        // Compute range with clamping to avoid division by zero
        float range = fmaxf(group_max - group_min, 1e-5f); 
        
        // Compute quantization factor (scale for 8-bit: 2^8 - 1 = 255)
        float q_factor_val = range / 255.0f;
        
        // Compute zero point with clamping to [0, 255]
        float zero_point_float = -roundf(group_min / q_factor_val);
        zero_point_float = fminf(fmaxf(zero_point_float, 0.0f), 255.0f);
        
        // Calculate group_id: which layer, output channel, and group
        int num_groups = CEIL_DIV(IC, GROUP_SIZE);
        int group_id = (l * OC + oc) * num_groups + blockIdx.x;
        
        // Store results
        q_factor[group_id] = q_factor_val;
        zero_point[group_id] = (uint8_t)zero_point_float;
    }
}

// Host function to launch calc_quant_param_kernel (batched - all layers)
void calc_quant_param(float *q_factor, uint8_t *zero_point, const float *weight, 
                              int L, int OC, int IC) {
    // 3D grid: (num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L) - all layers in parallel
    // 2D block: (GROUP_SIZE, OC_PER_BLOCK) = (128, 8) = 1024 threads
    int num_groups = CEIL_DIV(IC, GROUP_SIZE);
    dim3 grid(num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L);
    dim3 block(GROUP_SIZE, OC_PER_BLOCK);
    calc_quant_param_kernel<<<grid, block>>>(q_factor, zero_point, weight, L, OC, IC);
}

// ============================================================================
// HELPER DEVICE FUNCTION: QUANTIZE SINGLE VALUE
// ============================================================================

__device__ __forceinline__ uint8_t quantize_single_value(
    float weight,
    float q_factor,
    uint8_t zero_point)
{
    // Quantize: round(weight / q_factor) + zero_point
    float quantized_float = roundf(weight / q_factor) + (float)zero_point;
    
    // Clamp to [0, 255] for 8-bit range
    quantized_float = fminf(fmaxf(quantized_float, 0.0f), 255.0f);
    
    return (uint8_t)quantized_float;
}

// ============================================================================
// PSEUDO QUANTIZE KERNEL (QUANTIZE + IMMEDIATE DEQUANTIZE)
// ============================================================================

// Pseudo quantize weight matrix: quantize then immediately dequantize
// This simulates quantization error without actually storing packed values
// Useful for AWQ calibration and analysis
//
// Uses 3D block/thread indexing for batched processing:
//   - threadIdx.x (0-127): Position within a group
//   - threadIdx.y (0-7): Which output channel within the block
//   - blockIdx.x: Which group (along IC dimension)
//   - blockIdx.y: Which block of output channels
//   - blockIdx.z: Which layer
//
// Input:
//   - weights: (L, OC, IC) weight matrix (row-major)
//   - q_factors[(L, OC, num_groups)]: per-group scale factors
//   - zero_points[(L, OC, num_groups)]: per-group zero points
//   - L: number of layers
//   - OC: number of output channels (rows)
//   - IC: number of input channels (columns)
// Output:
//   - pseudo_quantized: (L, OC, IC) float matrix with quantization error
//
// Grid: dim3(num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L)
// Block: dim3(GROUP_SIZE, OC_PER_BLOCK) = (128, 8) = 1024 threads

#define OC_PER_BLOCK (QUANT_BLOCK_SIZE / GROUP_SIZE)  // = 8

__global__ void pseudo_quantize_weights_kernel(
    float* pseudo_quantized,
    const float* weights,
    const float* q_factors,
    const uint8_t* zero_points,
    int L, int OC, int IC, int num_groups)
{
    // 3D thread indexing
    int group_id = blockIdx.x;                              // Which group (0 to num_groups-1)
    int oc = blockIdx.y * OC_PER_BLOCK + threadIdx.y;       // Output channel (row)
    int ic = group_id * GROUP_SIZE + threadIdx.x;           // Input channel (column)
    int l = blockIdx.z;                                      // Layer index
    
    if (oc < OC && ic < IC && l < L) {
        // Load quantization parameters
        // All threads with same threadIdx.y access the same parameters
        int param_idx = (l * OC + oc) * num_groups + group_id;
        float q_factor = q_factors[param_idx];
        uint8_t zero_point = zero_points[param_idx];
        
        // Load weight value
        int weight_idx = (l * OC + oc) * IC + ic;
        float weight = weights[weight_idx];
        
        // Step 1: Quantize using existing helper function
        uint8_t quantized = quantize_single_value(weight, q_factor, zero_point);
        
        // Step 2: Dequantize - (q - zero_point) * q_factor
        float dequantized = ((float)quantized - (float)zero_point) * q_factor;
        
        // Store result
        pseudo_quantized[weight_idx] = dequantized;
    }
}

// Launcher function for pseudo quantization (batched - all layers)
void pseudo_quantize_weights(
    float* pseudo_quantized,
    const float* weights,
    const float* q_factors,        // Device pointer: (L, OC, num_groups)
    const uint8_t* zero_points,    // Device pointer: (L, OC, num_groups)
    int L, int OC, int IC)
{
    int num_groups = CEIL_DIV(IC, GROUP_SIZE);
    
    // 3D grid: (num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L) - all layers in parallel
    // 2D block: (GROUP_SIZE, OC_PER_BLOCK) = (128, 8) = 1024 threads
    dim3 grid(num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L);
    dim3 block(GROUP_SIZE, OC_PER_BLOCK);
    
    pseudo_quantize_weights_kernel<<<grid, block>>>(
        pseudo_quantized, weights, q_factors, zero_points, 
        L, OC, IC, num_groups);
}

// ============================================================================
// UINT32 PACKED UINT8 WEIGHT QUANTIZATION KERNEL (WITH AWQ SALIENT SCALING)
// ============================================================================

// Quantize weight matrix using per-group quantization parameters
// with AWQ salient channel scaling, packing 4 uint8_t values into uint32_t
// 
// Supports per-layer granularity with explicit layer indexing to match
// calc_quant_param_kernel(). When L=1, this processes a single layer.
// 
// Packing format: 
//   uint32 = (q0 << 0) | (q1 << 8) | (q2 << 16) | (q3 << 24)
//   - Byte 0: column col+0
//   - Byte 1: column col+1
//   - Byte 2: column col+2
//   - Byte 3: column col+3
// 
// Uses optimized 3D block/thread indexing with layer support:
//   - threadIdx.x (0-31): Which pack of 4 elements within a group (GROUP_SIZE/4 = 32)
//   - threadIdx.y (0-7): Which output channel within the block
//   - blockIdx.x: Which group (along IC dimension)
//   - blockIdx.y: Which block of output channels
//   - blockIdx.z: Layer index (0 to L-1)
//
// Quantization parameters (layer-aware indexing):
//   - q_factors[(L, OC, num_groups)]: scale factors
//   - zero_points[(L, OC, num_groups)]: zero points
//   - Layout: flattened 3D array, indexed as (l * OC + oc) * num_groups + group_id
//
// Input:
//   - weights: (L, OC, IC) weight matrix (row-major, batched across layers)
//   - L: number of layers
//   - OC: number of output channels (rows)
//   - IC: number of input channels (columns, must be divisible by 4)
// Output:
//   - quantized_packed: (L, OC, IC/4) packed uint32_t array
//
// Grid: dim3(num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L)
// Block: dim3(GROUP_SIZE/4, OC_PER_BLOCK) = (32, 8) = 256 threads

#define PACK_SIZE 4
#define PACKED_THREADS_PER_GROUP (GROUP_SIZE / PACK_SIZE)  // = 32

__global__ void quantize_weights_uint32_kernel(
    uint32_t* quantized_packed,
    const float* weights,
    const float* q_factors,
    const uint8_t* zero_points,
    int L, int OC, int IC, int num_groups)
{
    // 3D thread indexing
    int group_id = blockIdx.x;                              // Which group (0 to num_groups-1)
    int oc = blockIdx.y * OC_PER_BLOCK + threadIdx.y;       // Output channel (row)
    int l = blockIdx.z;                                      // Layer index
    int packed_idx = threadIdx.x;                           // Which pack of 4 within group (0-31)
    
    // Calculate base input channel for this thread's pack of 4
    int ic_base = group_id * GROUP_SIZE + packed_idx * PACK_SIZE;
    int col0 = ic_base;
    int col1 = ic_base + 1;
    int col2 = ic_base + 2;
    int col3 = ic_base + 3;
    
    if (oc < OC && col3 < IC && l < L) {
        // All 4 columns are in the same group, so we only need one set of parameters
        int param_idx = (l * OC + oc) * num_groups + group_id;
        float q_factor = q_factors[param_idx];
        uint8_t zero_point = zero_points[param_idx];
        
        // Load weight values from this layer
        int layer_offset = (l * OC + oc) * IC;
        int idx0 = layer_offset + col0;
        int idx1 = layer_offset + col1;
        int idx2 = layer_offset + col2;
        int idx3 = layer_offset + col3;
        
        float weight0 = weights[idx0];
        float weight1 = weights[idx1];
        float weight2 = weights[idx2];
        float weight3 = weights[idx3];
        
        // Quantize all 4 values using the same parameters (same group)
        uint8_t q0 = quantize_single_value(weight0, q_factor, zero_point);
        uint8_t q1 = quantize_single_value(weight1, q_factor, zero_point);
        uint8_t q2 = quantize_single_value(weight2, q_factor, zero_point);
        uint8_t q3 = quantize_single_value(weight3, q_factor, zero_point);
        
        // Pack four 8-bit values into one 32-bit word
        uint32_t packed = ((uint32_t)q0 << 0)  | 
                          ((uint32_t)q1 << 8)  | 
                          ((uint32_t)q2 << 16) | 
                          ((uint32_t)q3 << 24);
        
        // Store packed uint32_t
        int packed_col = group_id * PACKED_THREADS_PER_GROUP + packed_idx;
        int output_idx = (l * OC + oc) * (IC / PACK_SIZE) + packed_col;
        quantized_packed[output_idx] = packed;
    }
}

void quantize_weights_uint32(
    uint32_t* quantized_packed,
    const float* weights,
    const float* q_factors,        // Device pointer: (L, OC, num_groups)
    const uint8_t* zero_points,    // Device pointer: (L, OC, num_groups)
    int L, int OC, int IC)
{
    // Ensure IC is divisible by 4 (required for uint32_t packing)
    if (IC % 4 != 0) {
        fprintf(stderr, "Error: IC must be divisible by 4 for uint32_t packed quantization\n");
        return;
    }
    
    int num_groups = CEIL_DIV(IC, GROUP_SIZE);
    
    // Launch kernel with optimized 3D grid/block configuration
    // Grid: (num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L) - all layers in parallel
    // Block: (PACKED_THREADS_PER_GROUP, OC_PER_BLOCK) = (32, 8) = 256 threads
    dim3 grid(num_groups, CEIL_DIV(OC, OC_PER_BLOCK), L);
    dim3 block(PACKED_THREADS_PER_GROUP, OC_PER_BLOCK);
    
    quantize_weights_uint32_kernel<<<grid, block>>>(
        quantized_packed, weights, q_factors, zero_points, 
        L, OC, IC, num_groups);
}

#endif // __QUANTIZE_KERNEL_CUH__
