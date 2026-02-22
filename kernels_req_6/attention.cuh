#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <iostream>
using namespace nvcuda;
#define Q_TILE_SIZE 16
#define KV_TILE_SIZE 64    
#define WARP_SIZE 32
#define THREADS 128
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// ============================================================================
// CONSTANT MEMORY OPTIMIZATION (req_6)
// ============================================================================
// Store frequently accessed configuration parameters in constant memory
// Benefits:
// 1. Cached in constant cache (8KB per SM on modern GPUs)
// 2. Broadcast read - when all threads in a warp read the same address,
//    only one memory transaction is needed instead of 32
// 3. Reduces register pressure by not passing these as kernel parameters
// 4. Faster access than global memory for uniform reads
//
// Target: softmax_scale in flash attention kernel
// - Read by all threads in the attention kernel during QK^T scaling
// - Small size (4 bytes)
// - Perfect candidate for constant memory broadcast
// ============================================================================

__constant__ float c_softmax_scale; // 1/sqrt(head_dim)

__global__ void permute_kernel(float *q, float *k, float *v, const float *inp, int B, int N, int NH, int d) {
    // Implement this

    int b = blockIdx.x;
    int nh_idx = blockIdx.y;
    int t = threadIdx.x;

    if (t < N) {
        for (int i = 0; i < d; i++) {
            int inp_idx = (b * N * 3 * NH * d) + (t * 3 * NH * d) + (nh_idx * d) + i;
            int out_idx = (b * NH * N * d) + (nh_idx * N * d) + (t * d) + i;
            q[out_idx] = inp[inp_idx];
            k[out_idx] = inp[inp_idx + NH * d];
            v[out_idx] = inp[inp_idx + 2 * NH * d];
        }
    }
}

__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH, int d) {
    // Implement this

    int b = blockIdx.x;
    int nh_idx = blockIdx.y;
    int t = threadIdx.x;

    if (t < N) {
        for (int i = 0; i < d; i++) {
            int inp_idx = (b * NH * N * d) + (nh_idx * N * d) + (t * d) + i;
            int out_idx = (b * N * NH * d) + (t * NH * d) + (nh_idx * d) + i;
            out[out_idx] = inp[inp_idx];
        }
    }
}




__device__ void compute_qk_wmma(
    float* s_qK,              
    const __half* s_q,        
    const __half* s_K,        
    int D,                    
    int tid                   
) {
    int warp_id = tid / WARP_SIZE;

    if (D % WMMA_K == 0) {
        // Use WMMA for WMMA_M x WMMA_N tiles when D is multiple of WMMA_K
        if (warp_id < KV_TILE_SIZE / WMMA_M) { // Number of WMMA tiles needed
            int k_row_offset = warp_id * WMMA_M;
            int q_row_offset = 0; // Q tile starts at row 0

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            wmma::fill_fragment(acc_frag, 0.0f);

            for (int d = 0; d < D; d += WMMA_K) {
                wmma::load_matrix_sync(q_frag, s_q + q_row_offset * D + d, D);

                wmma::load_matrix_sync(k_frag, s_K + k_row_offset * D + d, D);

                wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
            }

            wmma::store_matrix_sync(
                s_qK + q_row_offset * KV_TILE_SIZE + k_row_offset,
                acc_frag,
                KV_TILE_SIZE,
                wmma::mem_row_major
            );
        }
    } else {
        // Fallback to standard loop for non-multiple-of-WMMA_K dimensions
        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {
            int kv_idx = qk_idx % KV_TILE_SIZE;
            int q_idx = qk_idx / KV_TILE_SIZE;

            float qk = 0.0f;
            for (int d = 0; d < D; d++) {
                qk += __half2float(s_q[q_idx * D + d]) * __half2float(s_K[kv_idx * D + d]);
            }
            s_qK[q_idx * KV_TILE_SIZE + kv_idx] = qk;
        }
    }
    __syncthreads();
}



__device__ void compute_pv_wmma(
    float* s_o,               
    const float* s_qK,        
    const __half* s_V,       
    int D,                    
    int tid                  
) {
    int warp_id = tid / WARP_SIZE;

    if (D % WMMA_K == 0) {
        __shared__ float s_P_tile[WMMA_M * WMMA_N];
        __shared__ __half s_P_half[WMMA_M * WMMA_N];
        __shared__ float s_pv_accum[WMMA_M * WMMA_K];

        int num_d_tiles = D / WMMA_K;

        for (int d_tile = 0; d_tile < num_d_tiles; d_tile++) {
            int d_col_offset = d_tile * WMMA_K;

            // Zero accumulator
            for (int i = tid; i < WMMA_M * WMMA_K; i += blockDim.x) {
                s_pv_accum[i] = 0.0f;
            }
            __syncthreads();

            // Accumulate across KV dimension in WMMA_N-element chunks
            for (int kv_chunk = 0; kv_chunk < KV_TILE_SIZE / WMMA_N; kv_chunk++) {
                int kv_offset = kv_chunk * WMMA_N;

                // Extract P tile (WMMA_M x WMMA_N) from s_qK
                for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
                    int row = i / WMMA_N;
                    int col = i % WMMA_N;
                    s_P_tile[i] = s_qK[row * KV_TILE_SIZE + kv_offset + col];
                }
                __syncthreads();

                // Convert P tile to half precision
                for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
                    s_P_half[i] = __float2half(s_P_tile[i]);
                }
                __syncthreads();

                if (warp_id == 0) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> P_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> V_frag;
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

                    wmma::load_matrix_sync(acc_frag, s_pv_accum, WMMA_K, wmma::mem_row_major);

                    wmma::load_matrix_sync(P_frag, s_P_half, WMMA_N);

                    wmma::load_matrix_sync(V_frag, s_V + kv_offset * D + d_col_offset, D);

                    wmma::mma_sync(acc_frag, P_frag, V_frag, acc_frag);

                    wmma::store_matrix_sync(s_pv_accum, acc_frag, WMMA_K, wmma::mem_row_major);
                }
                __syncthreads();
            }

            // Write accumulated tile to output
            for (int i = tid; i < WMMA_M * WMMA_K; i += blockDim.x) {
                int row = i / WMMA_K;
                int col = i % WMMA_K;
                s_o[row * D + d_col_offset + col] += s_pv_accum[i];
            }
            __syncthreads();
        }
    } else {
        // Fallback for non-multiple-of-WMMA_K dimensions
        for (int qd_idx = tid; qd_idx < Q_TILE_SIZE * D; qd_idx += blockDim.x) {
            int q_idx = qd_idx / D;
            int d_idx = qd_idx % D;
            float acc = 0.0f;
            for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
                acc += s_qK[q_idx * KV_TILE_SIZE + kv_idx] * __half2float(s_V[kv_idx * D + d_idx]);
            }
            s_o[qd_idx] += acc;
        }
    }
    __syncthreads();
}


__global__ void pv_matmul_no_transpose_kernel(float *out, const float *P, const float *V, int T, int d) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int hd = blockIdx.y * blockDim.y + threadIdx.y;

    if (t < T && hd < d) {
        float acc = 0.0f;
        for (int i = 0; i < T; i++) {
            acc += P[t * T + i] * V[i * d + hd];
        }
        out[t * d + hd] = acc;
    }
}



__global__ void flash_attn_kernel(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    int B, // num batchs
    int H, // num heads
    int T, // seq len
    int D  // head dim
    // Note: softmax_scale removed from parameters - now read from c_softmax_scale constant memory
) {
    int tid = threadIdx.x;
    // Each block will take reponsibility to compute output for one Q vector
    int batch_idx = blockIdx.z;
    int head_idx= blockIdx.y;
    int q_tile_offset = blockIdx.x * Q_TILE_SIZE;

    extern __shared__ char s_mem_raw[];
    __half* s_q = (__half*)s_mem_raw;
    __half* s_K = s_q + Q_TILE_SIZE * D;
    __half* s_V = s_K + KV_TILE_SIZE * D;
    float* s_qK = (float*)(s_V + KV_TILE_SIZE * D);
    float* s_o = s_qK + KV_TILE_SIZE * Q_TILE_SIZE;
    int batch_head_offset =  batch_idx * H * T * D + head_idx * T * D;
    __shared__ float s_block_max[Q_TILE_SIZE];
    __shared__ float s_sum_exp[Q_TILE_SIZE];
    __shared__ float s_prev_max[Q_TILE_SIZE];
    __shared__ float s_row_max[Q_TILE_SIZE];

    // init 0 for all qK and load current Q data into shared mem (convert to half)

    for (int d = tid; d < Q_TILE_SIZE * D; d += blockDim.x) {
        int q_idx = d / D;
        int d_idx = d % D;
        int q_offset = q_tile_offset + q_idx;
        if (q_offset < T) {
            s_q[d] = __float2half(Q[batch_head_offset + q_offset * D + d_idx]);
        } else {
            s_q[d] = __float2half(0.0f);
        }
        s_o[d] = 0.0f;
    }

    for (int t = tid; t < KV_TILE_SIZE * Q_TILE_SIZE; t += blockDim.x) {
        s_qK[t] = 0.0f;
    }

    for (int t = tid; t < Q_TILE_SIZE; t += blockDim.x) {
        s_block_max[t] = -INFINITY;
        s_row_max[t] = -INFINITY;
        s_sum_exp[t] = 0.0f;
    }

    __syncthreads(); 

    // For causal attention, only process tiles up to and including the one containing q_idx
    int num_tiles =  CEIL_DIV(q_tile_offset + Q_TILE_SIZE  + 1, KV_TILE_SIZE); 

    for (int tile = 0; tile < num_tiles; tile++) {
        // load all kv in current tile into shared mem (convert to half)
        for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
            int kv_offset = tile * KV_TILE_SIZE + kv_idx;
            if (kv_offset < T) {
                for (int d = tid; d < D; d += blockDim.x) {
                    s_K[kv_idx * D + d] = __float2half(K[batch_head_offset + kv_offset * D + d]);
                    s_V[kv_idx * D + d] = __float2half(V[batch_head_offset + kv_offset * D + d]);
                }
            } else {
                for (int d = tid; d < D; d += blockDim.x) {
                    s_K[kv_idx * D + d] = __float2half(0.0f);
                    s_V[kv_idx * D + d] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Compute Q @ K^T using WMMA or fallback
        compute_qk_wmma(s_qK, s_q, s_K, D, tid);

        // Apply softmax scaling and causal masking
        // CONSTANT MEMORY OPTIMIZATION: c_softmax_scale is read from constant memory
        // instead of being passed as a kernel parameter. This reduces parameter count
        // and benefits from constant cache broadcast when all threads read it.
        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {
            int kv_idx = qk_idx % KV_TILE_SIZE;
            int q_idx = qk_idx / KV_TILE_SIZE;
            int q_offset = q_tile_offset + q_idx;
            int kv_offset = tile * KV_TILE_SIZE + kv_idx;

            if (q_offset >= T || kv_offset >= T || kv_offset > q_offset) {
                s_qK[q_idx * KV_TILE_SIZE + kv_idx] = -INFINITY;
            } else {
                s_qK[q_idx * KV_TILE_SIZE + kv_idx] *= c_softmax_scale;  // Using constant memory
            }
        }
        __syncthreads();
        // find max in current block using shared memory reduction
        // TODO: reduction can be used here

        for (int q_idx = tid; q_idx < Q_TILE_SIZE; q_idx += blockDim.x) {
            float block_max = -INFINITY;
            for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
                block_max = fmaxf(block_max, s_qK[q_idx * KV_TILE_SIZE + kv_idx]);
            }
            s_block_max[q_idx] = block_max;
            s_prev_max[q_idx] = s_row_max[q_idx]; // Save previous max for rescaling
        }
        __syncthreads();

        // Update row max and compute exp
        for (int q_idx = tid; q_idx < Q_TILE_SIZE; q_idx += blockDim.x) {
            float new_max = fmaxf(s_block_max[q_idx], s_row_max[q_idx]);
            s_row_max[q_idx] = new_max;
        }
        __syncthreads();

        // Compute exp(qK - new_max)
        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {
            int kv_idx = qk_idx % KV_TILE_SIZE;
            int q_idx = qk_idx / KV_TILE_SIZE;
            s_qK[q_idx * KV_TILE_SIZE + kv_idx] = expf(s_qK[q_idx * KV_TILE_SIZE + kv_idx] - s_row_max[q_idx]);
        }
        __syncthreads();

        // Rescale previous o for each Q row
        for (int qd_idx = tid; qd_idx < Q_TILE_SIZE * D; qd_idx += blockDim.x) {
            int q_idx = qd_idx / D;
            float scale = expf(s_prev_max[q_idx] - s_row_max[q_idx]);
            s_o[qd_idx] = s_o[qd_idx] * scale;
        }
        __syncthreads();

        // Compute P @ V using WMMA or fallback
        compute_pv_wmma(s_o, s_qK, s_V, D, tid);

        // Compute and update sum_exp for each Q row
        for (int q_idx = tid; q_idx < Q_TILE_SIZE; q_idx += blockDim.x) {
            float block_sum_exp = 0.0f;
            for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
                block_sum_exp += s_qK[q_idx * KV_TILE_SIZE + kv_idx];
            }
            float scale = expf(s_prev_max[q_idx] - s_row_max[q_idx]);
            s_sum_exp[q_idx] = s_sum_exp[q_idx] * scale + block_sum_exp;
        }
        __syncthreads();
    }

    // write output
    for (int qd_idx = tid; qd_idx < Q_TILE_SIZE * D; qd_idx += blockDim.x) {
        int q_idx = qd_idx / D;
        int d_idx = qd_idx % D;
        int q_offset = q_tile_offset + q_idx;
        if (q_offset < T) {
            float sum_exp_inv = 1.0f / s_sum_exp[q_idx];
            O[batch_head_offset + q_offset * D + d_idx] = s_o[qd_idx] * sum_exp_inv;
        }
    }
}
        
        



// Launch all kernels related to attention here
void attention_forward(float *out, float *qkvr, float *att, float *inp, int B, int T, int C, int NH) {
    // Implement this

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // ============================================================================
    // CONSTANT MEMORY SETUP
    // Copy softmax scale to constant memory before kernel launch
    // This only happens once per attention_forward call, but the value is
    // cached and can be accessed efficiently by all threads in all blocks
    // ============================================================================
    float scale = 1.0f / sqrtf(HS);
    cudaMemcpyToSymbol(c_softmax_scale, &scale, sizeof(float));

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    dim3 permuteBlockDims(T);
    dim3 permuteGridDims(B, NH);
    permute_kernel<<<permuteGridDims, permuteBlockDims>>>(q, k, v, inp, B, T, NH, HS);
    cudaDeviceSynchronize();

    float *attention_out; 
    cudaMalloc((void**)&attention_out,  B * NH * T * HS * sizeof(float));

    dim3 num_threads(THREADS);
    dim3 num_blocks(CEIL_DIV(T, Q_TILE_SIZE), NH, B);

    // Dynamic shared memory:
    // s_q(Q_TILE_SIZE*HS half) + s_K(KV_TILE_SIZE*HS half) + s_V(KV_TILE_SIZE*HS half) + s_qK(Q_TILE_SIZE*KV_TILE_SIZE float) + s_o(Q_TILE_SIZE*HS float)
    size_t smem_size = (Q_TILE_SIZE * HS + KV_TILE_SIZE * HS + KV_TILE_SIZE * HS) * sizeof(__half) +
                       (Q_TILE_SIZE * KV_TILE_SIZE + Q_TILE_SIZE * HS) * sizeof(float);

    // Note: We no longer pass 'scale' as a parameter - it's read from constant memory
    flash_attn_kernel<<<num_blocks, num_threads, smem_size>>>(q, k, v, attention_out, B, NH, T, HS);
    cudaDeviceSynchronize();

    dim3 unpermuteBlockDims(T);
    dim3 unpermuteGridDims(B, NH);
    unpermute_kernel<<<unpermuteGridDims, unpermuteBlockDims>>>(attention_out, out, B, T, NH, HS);
    cudaDeviceSynchronize();

    cudaFree(attention_out);
}

#endif // __ATTENTION_CUH__
