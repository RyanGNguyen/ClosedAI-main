#ifndef __DEQUANTIZE_KERNEL_CUH__
#define __DEQUANTIZE_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

#define GROUP_SIZE 128
#define PACK_SIZE 4
#define WARP_SIZE 32

// Tile sizes for GEMM kernel
#define TILE_M 32  // Output rows per block
#define TILE_N 32  // Output cols per block
#define TILE_K 128 // Reduction dimension per iteration (must be multiple of GROUP_SIZE)

// Thread block configuration for GEMM
#define THREADS_PER_BLOCK 256
#define THREADS_M 16
#define THREADS_N 16

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

// Dequantize a single uint8 value to float
// Formula: dequantized = (quantized - zero_point) * q_factor
__device__ __forceinline__ float dequantize_single_value(
    uint8_t quantized,
    uint8_t zero_point,
    float q_factor)
{
    return ((float)quantized - (float)zero_point) * q_factor;
}

// Unpack a uint32_t containing 4 packed uint8 values and dequantize them
// Packing format: uint32 = (q0 << 0) | (q1 << 8) | (q2 << 16) | (q3 << 24)
// Output: 4 consecutive floats in dequant_out[0..3]
__device__ __forceinline__ void unpack_and_dequantize_uint32(
    float* dequant_out,
    uint32_t packed,
    uint8_t zero_point,
    float q_factor)
{
    // Extract 4 uint8 values from packed uint32
    uint8_t q0 = (packed >> 0)  & 0xFF;
    uint8_t q1 = (packed >> 8)  & 0xFF;
    uint8_t q2 = (packed >> 16) & 0xFF;
    uint8_t q3 = (packed >> 24) & 0xFF;
    
    // Dequantize all 4 values
    dequant_out[0] = dequantize_single_value(q0, zero_point, q_factor);
    dequant_out[1] = dequantize_single_value(q1, zero_point, q_factor);
    dequant_out[2] = dequantize_single_value(q2, zero_point, q_factor);
    dequant_out[3] = dequantize_single_value(q3, zero_point, q_factor);
}

// ============================================================================
// STANDALONE DEQUANTIZE KERNEL (FOR VALIDATION/TESTING)
// ============================================================================

// Dequantize packed uint32 weights to full float32 weights
// Input:
//   - d_quantized_packed: (OC, IC/4) packed uint32 array
//   - d_q_factors: (OC, num_groups) quantization factors
//   - d_zero_points: (OC, num_groups) zero points
//   - OC: number of output channels
//   - IC: number of input channels (must be divisible by 4)
// Output:
//   - d_weights: (OC, IC) float32 array
__global__ void dequantize_weights_kernel(
    float* d_weights,
    const uint32_t* d_quantized_packed,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    int OC, int IC, int num_groups)
{
    // Each thread processes one packed uint32 (4 consecutive IC values)
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int packed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ic_base = packed_idx * PACK_SIZE;
    
    if (oc < OC && ic_base < IC) {
        // Calculate which group this packed value belongs to
        int group_id = ic_base / GROUP_SIZE;
        int param_idx = oc * num_groups + group_id;
        
        // Load quantization parameters
        float q_factor = d_q_factors[param_idx];
        uint8_t zero_point = d_zero_points[param_idx];
        
        // Load packed value
        int packed_col = ic_base / PACK_SIZE;
        uint32_t packed = d_quantized_packed[oc * (IC / PACK_SIZE) + packed_col];
        
        // Dequantize to 4 floats
        float dequant[4];
        unpack_and_dequantize_uint32(dequant, packed, zero_point, q_factor);
        
        // Store to global memory
        int out_idx = oc * IC + ic_base;
        if (ic_base + 3 < IC) {
            d_weights[out_idx + 0] = dequant[0];
            d_weights[out_idx + 1] = dequant[1];
            d_weights[out_idx + 2] = dequant[2];
            d_weights[out_idx + 3] = dequant[3];
        } else {
            // Handle edge case where IC is not perfectly divisible by 4
            for (int i = 0; i < 4 && (ic_base + i) < IC; i++) {
                d_weights[out_idx + i] = dequant[i];
            }
        }
    }
}

// ============================================================================
// FUSED GEMV KERNEL (OPTIMIZED FOR SMALL BATCH SIZES)
// ============================================================================

// Compute: Y = (X / s) @ W_dequant^T + bias
// Optimized for B*T <= 8 (common in inference)
// 
// Input:
//   - d_inp: (B*T, IC) input activations
//   - d_weight_packed: (OC, IC/4) packed quantized weights
//   - d_scales: (IC,) per-channel activation scales
//   - d_q_factors: (OC, num_groups) weight quantization factors
//   - d_zero_points: (OC, num_groups) weight zero points
//   - d_bias: (OC,) bias vector (can be NULL)
//   - B, T, IC, OC: dimensions
// Output:
//   - d_out: (B*T, OC) output
//
// Grid: (OC, B*T)
// Block: 256 threads (8 warps)
__global__ void fused_dequant_scale_gemv_kernel(
    float* d_out,
    const float* d_inp,
    const uint32_t* d_weight_packed,
    const float* d_scales,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    const float* d_bias,
    int B, int T, int IC, int OC, int num_groups)
{
    int oc = blockIdx.x;  // Output channel
    int bt = blockIdx.y;  // Batch-time index
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    if (oc >= OC || bt >= B * T) return;
    
    // Each thread accumulates a partial dot product
    float partial_sum = 0.0f;
    
    // Process IC dimension in chunks of PACK_SIZE (4)
    int num_packed = IC / PACK_SIZE;
    
    for (int packed_idx = tid; packed_idx < num_packed; packed_idx += num_threads) {
        int ic_base = packed_idx * PACK_SIZE;
        
        // Determine which group for quantization parameters
        int group_id = ic_base / GROUP_SIZE;
        int param_idx = oc * num_groups + group_id;
        
        float q_factor = d_q_factors[param_idx];
        uint8_t zero_point = d_zero_points[param_idx];
        
        // Load packed weight
        uint32_t packed = d_weight_packed[oc * num_packed + packed_idx];
        
        // Dequantize
        float weight[4];
        unpack_and_dequantize_uint32(weight, packed, zero_point, q_factor);
        
        // Load and scale input activations, then accumulate
        int inp_idx = bt * IC + ic_base;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int ic = ic_base + i;
            if (ic < IC) {
                float inp_val = d_inp[inp_idx + i];
                float scale = d_scales[ic];
                float scaled_inp = inp_val / scale;
                partial_sum += scaled_inp * weight[i];
            }
        }
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
    }
    
    // Use shared memory for inter-warp reduction
    __shared__ float smem[8];  // 8 warps max
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (lane_id == 0) {
        smem[warp_id] = partial_sum;
    }
    __syncthreads();
    
    // First warp does final reduction
    if (warp_id == 0) {
        int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
        float final_sum = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
        }
        
        // Thread 0 writes the result
        if (tid == 0) {
            if (d_bias != NULL) {
                final_sum += d_bias[oc];
            }
            d_out[bt * OC + oc] = final_sum;
        }
    }
}

// ============================================================================
// FUSED GEMM KERNEL (OPTIMIZED FOR LARGER BATCH SIZES)
// ============================================================================

// Compute: Y = (X / s) @ W_dequant^T + bias
// Tiled implementation with shared memory for larger batches
//
// Input:
//   - d_inp: (B*T, IC) input activations
//   - d_weight_packed: (OC, IC/4) packed quantized weights
//   - d_scales: (IC,) per-channel activation scales
//   - d_q_factors: (OC, num_groups) weight quantization factors
//   - d_zero_points: (OC, num_groups) weight zero points
//   - d_bias: (OC,) bias vector (can be NULL)
//   - B, T, IC, OC: dimensions
// Output:
//   - d_out: (B*T, OC) output
//
// Grid: (ceil(OC/TILE_N), ceil(B*T/TILE_M))
// Block: (THREADS_N, THREADS_M) = (16, 16) = 256 threads
__global__ void fused_dequant_scale_gemm_kernel(
    float* d_out,
    const float* d_inp,
    const uint32_t* d_weight_packed,
    const float* d_scales,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    const float* d_bias,
    int B, int T, int IC, int OC, int num_groups)
{
    // Block indices
    int block_row = blockIdx.y;  // Which tile of B*T
    int block_col = blockIdx.x;  // Which tile of OC
    
    // Thread indices within block
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Global row and column this thread is responsible for
    int global_row = block_row * TILE_M + thread_row;
    int global_col = block_col * TILE_N + thread_col;
    
    // Shared memory for tiles
    __shared__ float smem_inp[TILE_M][TILE_K];
    __shared__ float smem_weight[TILE_N][TILE_K];
    
    // Accumulator for this thread's output element
    float acc = 0.0f;
    
    int num_packed = IC / PACK_SIZE;
    int num_tiles = (IC + TILE_K - 1) / TILE_K;
    
    // Loop over tiles along IC dimension
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int k_start = tile_idx * TILE_K;
        
        // Load input tile (with scaling) into shared memory
        // Each thread loads multiple elements
        for (int k = thread_col; k < TILE_K; k += THREADS_N) {
            int ic = k_start + k;
            if (global_row < B * T && ic < IC) {
                float inp_val = d_inp[global_row * IC + ic];
                float scale = d_scales[ic];
                smem_inp[thread_row][k] = inp_val / scale;
            } else {
                smem_inp[thread_row][k] = 0.0f;
            }
        }
        
        // Load weight tile (with dequantization) into shared memory
        // Each thread loads and dequantizes multiple packed values
        for (int k = thread_row * PACK_SIZE; k < TILE_K; k += THREADS_M * PACK_SIZE) {
            int ic = k_start + k;
            if (global_col < OC && ic < IC) {
                int packed_idx = ic / PACK_SIZE;
                int group_id = ic / GROUP_SIZE;
                int param_idx = global_col * num_groups + group_id;
                
                float q_factor = d_q_factors[param_idx];
                uint8_t zero_point = d_zero_points[param_idx];
                
                uint32_t packed = d_weight_packed[global_col * num_packed + packed_idx];
                
                float weight[4];
                unpack_and_dequantize_uint32(weight, packed, zero_point, q_factor);
                
                // Store dequantized values (transposed)
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    if ((k + i) < TILE_K && (ic + i) < IC) {
                        smem_weight[thread_col][k + i] = weight[i];
                    }
                }
            } else {
                // Out of bounds - zero out
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    if ((k + i) < TILE_K) {
                        smem_weight[thread_col][k + i] = 0.0f;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            acc += smem_inp[thread_row][k] * smem_weight[thread_col][k];
        }
        
        __syncthreads();
    }
    
    // Write output
    if (global_row < B * T && global_col < OC) {
        if (d_bias != NULL) {
            acc += d_bias[global_col];
        }
        d_out[global_row * OC + global_col] = acc;
    }
}

// ============================================================================
// HOST LAUNCHER FUNCTIONS
// ============================================================================

// Standalone dequantization: unpack and dequantize weights
void dequantize_weights(
    float* d_weights,
    const uint32_t* d_quantized_packed,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    int OC, int IC)
{
    if (IC % 4 != 0) {
        fprintf(stderr, "Error: IC must be divisible by 4 for packed dequantization\n");
        return;
    }
    
    int num_groups = CEIL_DIV(IC, GROUP_SIZE);
    int num_packed = IC / PACK_SIZE;
    
    // Grid: (num_packed_cols, OC)
    // Block: (min(256, num_packed), 1)
    dim3 block(min(256, num_packed), 1);
    dim3 grid(CEIL_DIV(num_packed, block.x), OC);
    
    dequantize_weights_kernel<<<grid, block>>>(
        d_weights, d_quantized_packed, d_q_factors, d_zero_points,
        OC, IC, num_groups);
    
    cudaCheck(cudaGetLastError());
}

// Fused dequantize + scale + matmul (without bias)
void fused_dequant_scale_matmul(
    float* d_out,
    const float* d_inp,
    const uint32_t* d_weight_packed,
    const float* d_scales,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    int B, int T, int IC, int OC)
{
    if (IC % 4 != 0) {
        fprintf(stderr, "Error: IC must be divisible by 4 for packed weights\n");
        return;
    }
    
    int num_groups = CEIL_DIV(IC, GROUP_SIZE);
    int BT = B * T;
    
    // Choose kernel based on batch size
    if (BT <= 8) {
        // Use GEMV kernel for small batches
        dim3 grid(OC, BT);
        dim3 block(256);
        
        fused_dequant_scale_gemv_kernel<<<grid, block>>>(
            d_out, d_inp, d_weight_packed, d_scales,
            d_q_factors, d_zero_points, NULL,
            B, T, IC, OC, num_groups);
    } else {
        // Use GEMM kernel for larger batches
        dim3 grid(CEIL_DIV(OC, TILE_N), CEIL_DIV(BT, TILE_M));
        dim3 block(THREADS_N, THREADS_M);
        
        fused_dequant_scale_gemm_kernel<<<grid, block>>>(
            d_out, d_inp, d_weight_packed, d_scales,
            d_q_factors, d_zero_points, NULL,
            B, T, IC, OC, num_groups);
    }
    
    cudaCheck(cudaGetLastError());
}

// Fused dequantize + scale + matmul + bias
void fused_dequant_scale_matmul_bias(
    float* d_out,
    const float* d_inp,
    const uint32_t* d_weight_packed,
    const float* d_scales,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    const float* d_bias,
    int B, int T, int IC, int OC)
{
    if (IC % 4 != 0) {
        fprintf(stderr, "Error: IC must be divisible by 4 for packed weights\n");
        return;
    }
    
    int num_groups = CEIL_DIV(IC, GROUP_SIZE);
    int BT = B * T;
    
    // Choose kernel based on batch size
    if (BT <= 8) {
        // Use GEMV kernel for small batches
        dim3 grid(OC, BT);
        dim3 block(256);
        
        fused_dequant_scale_gemv_kernel<<<grid, block>>>(
            d_out, d_inp, d_weight_packed, d_scales,
            d_q_factors, d_zero_points, d_bias,
            B, T, IC, OC, num_groups);
    } else {
        // Use GEMM kernel for larger batches
        dim3 grid(CEIL_DIV(OC, TILE_N), CEIL_DIV(BT, TILE_M));
        dim3 block(THREADS_N, THREADS_M);
        
        fused_dequant_scale_gemm_kernel<<<grid, block>>>(
            d_out, d_inp, d_weight_packed, d_scales,
            d_q_factors, d_zero_points, d_bias,
            B, T, IC, OC, num_groups);
    }
    
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// CUBLAS-ACCELERATED FUNCTIONS
// ============================================================================

// Kernel to scale input activations: out = inp / scales
// Each thread processes one element
__global__ void scale_input_kernel(
    float* out,
    const float* inp,
    const float* scales,
    int rows,
    int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = inp[idx] / scales[col];
    }
}

// Kernel to add bias to output matrix
// Each thread processes one element
__global__ void add_bias_kernel(
    float* out,
    const float* bias,
    int rows,
    int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] += bias[col];
    }
}

// cuBLAS-accelerated fused dequantize + scale + matmul (without bias)
// Uses pre-allocated temporary buffers to avoid repeated allocations
//
// Input:
//   - d_inp: (B*T, IC) input activations
//   - d_weight_packed: (OC, IC/4) packed quantized weights
//   - d_scales: (IC,) per-channel activation scales
//   - d_q_factors: (OC, num_groups) weight quantization factors
//   - d_zero_points: (OC, num_groups) weight zero points
//   - d_weight_temp: Pre-allocated temp buffer for dequantized weights (OC, IC)
//   - d_inp_scaled_temp: Pre-allocated temp buffer for scaled input (B*T, IC)
//   - B, T, IC, OC: dimensions
// Output:
//   - d_out: (B*T, OC) output
void fused_dequant_scale_matmul_cublas(
    float* d_out,
    const float* d_inp,
    const uint32_t* d_weight_packed,
    const float* d_scales,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    float* d_weight_temp,
    float* d_inp_scaled_temp,
    int B, int T, int IC, int OC)
{
    if (IC % 4 != 0) {
        fprintf(stderr, "Error: IC must be divisible by 4 for packed weights\n");
        return;
    }
    
    int BT = B * T;
    
    // Step 1: Scale input activations: d_inp_scaled_temp = d_inp / d_scales
    dim3 block_scale(16, 16);
    dim3 grid_scale(CEIL_DIV(IC, block_scale.x), CEIL_DIV(BT, block_scale.y));
    scale_input_kernel<<<grid_scale, block_scale>>>(
        d_inp_scaled_temp, d_inp, d_scales, BT, IC);
    cudaCheck(cudaGetLastError());
    
    // Step 2: Dequantize weights to temp buffer
    dequantize_weights(d_weight_temp, d_weight_packed, d_q_factors, d_zero_points, OC, IC);
    
    // Step 3: cuBLAS GEMM: out = scaled_inp @ weight^T
    // We want: out = inp_scaled @ weight^T (row-major)
    // Equivalently: out^T = weight @ inp_scaled^T (column-major)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int m = OC;      // rows of result (column-major)
    int n = BT;      // cols of result (column-major)
    int k = IC;      // inner dimension
    
    cublasCheck(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,              // transpose weight^T to get weight
        CUBLAS_OP_N,              // use inp_scaled^T as-is
        m, n, k,                  // dimensions
        &alpha,
        d_weight_temp, k,         // A matrix (weight^T in memory, IC×OC, lda=IC)
        d_inp_scaled_temp, k,     // B matrix (inp_scaled^T in memory, IC×BT, lda=IC)
        &beta,
        d_out, m                  // C matrix (out^T in memory, OC×BT, lda=OC)
    ));
}

// cuBLAS-accelerated fused dequantize + scale + matmul + bias
// Uses pre-allocated temporary buffers to avoid repeated allocations
//
// Input:
//   - d_inp: (B*T, IC) input activations
//   - d_weight_packed: (OC, IC/4) packed quantized weights
//   - d_scales: (IC,) per-channel activation scales
//   - d_q_factors: (OC, num_groups) weight quantization factors
//   - d_zero_points: (OC, num_groups) weight zero points
//   - d_bias: (OC,) bias vector
//   - d_weight_temp: Pre-allocated temp buffer for dequantized weights (OC, IC)
//   - d_inp_scaled_temp: Pre-allocated temp buffer for scaled input (B*T, IC)
//   - B, T, IC, OC: dimensions
// Output:
//   - d_out: (B*T, OC) output
void fused_dequant_scale_matmul_bias_cublas(
    float* d_out,
    const float* d_inp,
    const uint32_t* d_weight_packed,
    const float* d_scales,
    const float* d_q_factors,
    const uint8_t* d_zero_points,
    const float* d_bias,
    float* d_weight_temp,
    float* d_inp_scaled_temp,
    int B, int T, int IC, int OC)
{
    // First do the matmul without bias
    fused_dequant_scale_matmul_cublas(
        d_out, d_inp, d_weight_packed, d_scales,
        d_q_factors, d_zero_points,
        d_weight_temp, d_inp_scaled_temp,
        B, T, IC, OC);
    
    // Step 4: Add bias
    int BT = B * T;
    dim3 block_bias(16, 16);
    dim3 grid_bias(CEIL_DIV(OC, block_bias.x), CEIL_DIV(BT, block_bias.y));
    add_bias_kernel<<<grid_bias, block_bias>>>(d_out, d_bias, BT, OC);
    cudaCheck(cudaGetLastError());
}

#endif // __DEQUANTIZE_KERNEL_CUH__
