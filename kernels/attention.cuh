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

// TF32 WMMA dimensions: M=16, N=16, K=8
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;  

__global__ void permute_kernel(
    float *q, 
    float *k, 
    float *v, 
    const float *__restrict__ inp, int B, int N, int NH, int d) {
    int b = blockIdx.x;
    int nh_idx = blockIdx.y;
    
    // Use grid-stride loop instead of assuming one thread per token
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        for (int i = 0; i < d; i++) {
            int inp_idx = (b * N * 3 * NH * d) + (t * 3 * NH * d) + (nh_idx * d) + i;
            int out_idx = (b * NH * N * d) + (nh_idx * N * d) + (t * d) + i;
            q[out_idx] = inp[inp_idx];
            k[out_idx] = inp[inp_idx + NH * d];
            v[out_idx] = inp[inp_idx + 2 * NH * d];
        }
    }
}

__global__ void unpermute_kernel(float *__restrict__ inp, float *out, int B, int N, int NH, int d) {
    int b = blockIdx.x;
    int nh_idx = blockIdx.y;
    
    // Use grid-stride loop instead of assuming one thread per token
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        for (int i = 0; i < d; i++) {
            int inp_idx = (b * NH * N * d) + (nh_idx * N * d) + (t * d) + i;
            int out_idx = (b * N * NH * d) + (t * NH * d) + (nh_idx * d) + i;
            out[out_idx] = inp[inp_idx];
        }
    }
}

__device__ void debug(int tid, int value) {
    if (tid == 0) printf("%d",value);
}
__device__ void compute_qk_wmma(
    float* s_qK,
    const float* s_q,  
    const float* s_K,  
    int D,
    int tid
) {
    int warp_id = tid / WARP_SIZE;

    // TF32 requires D % 8 == 0 (WMMA_K = 8)
    if (D % WMMA_K == 0 && KV_TILE_SIZE % WMMA_N == 0 && Q_TILE_SIZE % WMMA_M == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        int num_k_tiles = KV_TILE_SIZE / WMMA_N;

        for (int k_tile = warp_id; k_tile < num_k_tiles; k_tile += num_warps) {
            int k_row_offset = k_tile * WMMA_N;

            // TF32 fragments - use wmma::precision::tf32 for input matrices
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            wmma::fill_fragment(acc_frag, 0.0f);

            // Loop over D dimension in chunks of WMMA_K (8 for TF32)
            for (int d = 0; d < D; d += WMMA_K) {
                // Load Q tile: [Q_TILE_SIZE, WMMA_K] from s_q
                wmma::load_matrix_sync(q_frag, s_q + d, D);
                // Load K tile: [WMMA_N, WMMA_K] from s_K (col-major = transposed)
                wmma::load_matrix_sync(k_frag, s_K + k_row_offset * D + d, D);
                wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
            }

            wmma::store_matrix_sync(
                s_qK + k_row_offset,
                acc_frag,
                KV_TILE_SIZE,
                wmma::mem_row_major
            );
        }
    } else {
        // Fallback for non-aligned dimensions
        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {
            int kv_idx = qk_idx % KV_TILE_SIZE;
            int q_idx = qk_idx / KV_TILE_SIZE;

            float qk = 0.0f;
            for (int d = 0; d < D; d++) {
                qk += s_q[q_idx * D + d] * s_K[kv_idx * D + d];
            }
            s_qK[q_idx * KV_TILE_SIZE + kv_idx] = qk;
        }
    }
    __syncthreads();
}

__device__ void compute_pv_wmma(
    float* s_o,
    const float* s_qK, 
    const float* s_V,   
    int D,
    int tid
) {
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // P @ V: [Q_TILE_SIZE, KV_TILE_SIZE] @ [KV_TILE_SIZE, D] = [Q_TILE_SIZE, D]
    if (D % WMMA_N == 0 && KV_TILE_SIZE % WMMA_K == 0 && Q_TILE_SIZE % WMMA_M == 0) {
        int num_d_tiles = D / WMMA_N;

        for (int d_tile = warp_id; d_tile < num_d_tiles; d_tile += num_warps) {
            int d_offset = d_tile * WMMA_N;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            // Load existing output for accumulation
            wmma::load_matrix_sync(acc_frag, s_o + d_offset, D, wmma::mem_row_major);

            // Loop over KV dimension in chunks of WMMA_K (8)
            for (int kv = 0; kv < KV_TILE_SIZE; kv += WMMA_K) {
                wmma::load_matrix_sync(p_frag, s_qK + kv, KV_TILE_SIZE);
                wmma::load_matrix_sync(v_frag, s_V + kv * D + d_offset, D);
                wmma::mma_sync(acc_frag, p_frag, v_frag, acc_frag);
            }

            wmma::store_matrix_sync(s_o + d_offset, acc_frag, D, wmma::mem_row_major);
        }
    } else {
        // Fallback implementation
        int lane_id = tid % WARP_SIZE;
        
        for (int d_start = warp_id * 16; d_start < D; d_start += num_warps * 16) {
            int d_end = min(d_start + 16, D);

            for (int q_row = lane_id; q_row < Q_TILE_SIZE; q_row += WARP_SIZE) {
                for (int d_col = 0; d_col < (d_end - d_start); d_col++) {
                    float acc = 0.0f;
                    #pragma unroll 4
                    for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
                        float p_val = s_qK[q_row * KV_TILE_SIZE + kv_idx];
                        float v_val = s_V[kv_idx * D + d_start + d_col];
                        acc += p_val * v_val;
                    }
                    s_o[q_row * D + d_start + d_col] += acc;
                }
            }
        }
    }
    __syncthreads();
}

__global__ void flash_attn_kernel_tf32(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int B,
    int H,
    int T,
    int D,
    float softmax_scale
) {
    int tid = threadIdx.x;
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_tile_offset = blockIdx.x * Q_TILE_SIZE;

    extern __shared__ char s_mem_raw[];
    
    // All shared memory now uses float (TF32 operates on float inputs)
    float* s_q = (float*)s_mem_raw;
    float* s_K = s_q + Q_TILE_SIZE * D;
    float* s_V = s_K + KV_TILE_SIZE * D;
    float* s_qK = s_V + KV_TILE_SIZE * D;
    float* s_o = s_qK + KV_TILE_SIZE * Q_TILE_SIZE;
    
    int batch_head_offset = batch_idx * H * T * D + head_idx * T * D;
    
    __shared__ float s_block_max[Q_TILE_SIZE];
    __shared__ float s_sum_exp[Q_TILE_SIZE];
    __shared__ float s_prev_max[Q_TILE_SIZE];
    __shared__ float s_row_max[Q_TILE_SIZE];

    for (int d = tid; d < Q_TILE_SIZE * D; d += blockDim.x) {
        int q_idx = d / D;
        int d_idx = d % D;
        int q_offset = q_tile_offset + q_idx;
        if (q_offset < T) {
            s_q[d] = Q[batch_head_offset + q_offset * D + d_idx];
        } else {
            s_q[d] = 0.0f;
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

    int num_tiles = CEIL_DIV(q_tile_offset + Q_TILE_SIZE + 1, KV_TILE_SIZE);

    for (int tile = 0; tile < num_tiles; tile++) {
        for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
            int kv_offset = tile * KV_TILE_SIZE + kv_idx;
            if (kv_offset < T) {
                for (int d = tid; d < D; d += blockDim.x) {
                    s_K[kv_idx * D + d] = K[batch_head_offset + kv_offset * D + d];
                    s_V[kv_idx * D + d] = V[batch_head_offset + kv_offset * D + d];
                }
            } else {
                for (int d = tid; d < D; d += blockDim.x) {
                    s_K[kv_idx * D + d] = 0.0f;
                    s_V[kv_idx * D + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute Q @ K^T using TF32 WMMA
        compute_qk_wmma(s_qK, s_q, s_K, D, tid);

        // Apply softmax scaling and causal masking
        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {
            int kv_idx = qk_idx % KV_TILE_SIZE;
            int q_idx = qk_idx / KV_TILE_SIZE;
            int q_offset = q_tile_offset + q_idx;
            int kv_offset = tile * KV_TILE_SIZE + kv_idx;

            if (q_offset >= T || kv_offset >= T || kv_offset > q_offset) {
                s_qK[q_idx * KV_TILE_SIZE + kv_idx] = -INFINITY;
            } else {
                s_qK[q_idx * KV_TILE_SIZE + kv_idx] *= softmax_scale;
            }
        }
        __syncthreads();

        // Find max in current block
        for (int q_idx = tid; q_idx < Q_TILE_SIZE; q_idx += blockDim.x) {
            float block_max = -INFINITY;
            for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
                block_max = fmaxf(block_max, s_qK[q_idx * KV_TILE_SIZE + kv_idx]);
            }
            s_block_max[q_idx] = block_max;
            s_prev_max[q_idx] = s_row_max[q_idx];
        }
        __syncthreads();

        // Update row max
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

        // Rescale previous output
        for (int qd_idx = tid; qd_idx < Q_TILE_SIZE * D; qd_idx += blockDim.x) {
            int q_idx = qd_idx / D;
            float scale = expf(s_prev_max[q_idx] - s_row_max[q_idx]);
            s_o[qd_idx] = s_o[qd_idx] * scale;
        }
        __syncthreads();

        // Compute P @ V using TF32 WMMA
        compute_pv_wmma(s_o, s_qK, s_V, D, tid);

        // Update sum_exp
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

    // Write output
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

void attention_forward(float *out, float *qkvr, float *att, float *inp, int B, int T, int C, int NH) {
    int HS = C / NH;

    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    
    // Fix: Use fixed number of threads, not T
    int permute_threads = min(256, T);  // Or 128, 256, etc.
    dim3 permuteBlockDims(permute_threads);
    dim3 permuteGridDims(B, NH);
    permute_kernel<<<permuteGridDims, permuteBlockDims>>>(q, k, v, inp, B, T, NH, HS);
    cudaDeviceSynchronize();

    float *attention_out;
    cudaMalloc((void**)&attention_out, B * NH * T * HS * sizeof(float));

    dim3 num_threads(THREADS);
    dim3 num_blocks(CEIL_DIV(T, Q_TILE_SIZE), NH, B);
    float scale = 1.0f / sqrtf((float)HS);
    
    size_t smem_size = (Q_TILE_SIZE * HS + KV_TILE_SIZE * HS + KV_TILE_SIZE * HS) * sizeof(float) +
                       (Q_TILE_SIZE * KV_TILE_SIZE + Q_TILE_SIZE * HS) * sizeof(float);
    
    flash_attn_kernel_tf32<<<num_blocks, num_threads, smem_size>>>(q, k, v, attention_out, B, NH, T, HS, scale);
    cudaDeviceSynchronize();

    // Fix: Use fixed number of threads, not T
    int unpermute_threads = min(256, T);
    dim3 unpermuteBlockDims(unpermute_threads);
    dim3 unpermuteGridDims(B, NH);
    unpermute_kernel<<<unpermuteGridDims, unpermuteBlockDims>>>(attention_out, out, B, T, NH, HS);
    cudaDeviceSynchronize();

    cudaFree(attention_out);
}

#endif // __ATTENTION_CUH__