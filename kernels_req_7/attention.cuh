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

__global__ void permute_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const float* __restrict__ inp,
    int B, int N, int NH, int d) {

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


__global__ void unpermute_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int B, int N, int NH, int d) {

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
    float* __restrict__ s_qK,
    const __half* __restrict__ s_q,
    const __half* __restrict__ s_K,
    int D,
    int tid) {
    
    int warp_id = tid / WARP_SIZE;

    if (D % WMMA_K == 0) {
        // Use WMMA for WMMA_M x WMMA_N tiles when D is multiple of WMMA_K
        if (warp_id < KV_TILE_SIZE / WMMA_M) {
            int k_row_offset = warp_id * WMMA_M;
            int q_row_offset = 0;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
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

/**
 * Compute P @ V (attention weights times values)
 */
__device__ void compute_pv_wmma(
    float* __restrict__ s_o,
    const float* __restrict__ s_qK,
    const __half* __restrict__ s_V,
    int D,
    int tid) {
    
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

/**
 * P @ V matrix multiplication kernel (not transposed)
 */
__global__ void pv_matmul_no_transpose_kernel(
    float* __restrict__ out,
    const float* __restrict__ P,
    const float* __restrict__ V,
    int T, int d) {

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
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B,
    int H,
    int T,
    int D,
    float softmax_scale) {
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_tile_offset = blockIdx.x * Q_TILE_SIZE;

    extern __shared__ char s_mem_raw[];
    __half* s_q = (__half*)s_mem_raw;
    __half* s_K = s_q + Q_TILE_SIZE * D;
    __half* s_V = s_K + KV_TILE_SIZE * D;
    float* s_qK = (float*)(s_V + KV_TILE_SIZE * D);
    float* s_o = s_qK + KV_TILE_SIZE * Q_TILE_SIZE;
    
    int batch_head_offset = batch_idx * H * T * D + head_idx * T * D;
    
    __shared__ float s_block_max[Q_TILE_SIZE];
    __shared__ float s_sum_exp[Q_TILE_SIZE];
    __shared__ float s_prev_max[Q_TILE_SIZE];
    __shared__ float s_row_max[Q_TILE_SIZE];

    // Initialize shared memory and load Q tile
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
    int num_tiles = CEIL_DIV(q_tile_offset + Q_TILE_SIZE + 1, KV_TILE_SIZE);

    for (int tile = 0; tile < num_tiles; tile++) {
        // Load K and V tiles into shared memory
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

        // Rescale previous output for each Q row
        for (int qd_idx = tid; qd_idx < Q_TILE_SIZE * D; qd_idx += blockDim.x) {
            int q_idx = qd_idx / D;
            float scale = expf(s_prev_max[q_idx] - s_row_max[q_idx]);
            s_o[qd_idx] = s_o[qd_idx] * scale;
        }
        __syncthreads();

        // Compute P @ V using WMMA or fallback
        compute_pv_wmma(s_o, s_qK, s_V, D, tid);

        // Update sum_exp for each Q row
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

    // Write output with normalization
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

void attention_forward(
    float* __restrict__ out,
    float* __restrict__ qkvr,
    float* __restrict__ att,
    float* __restrict__ inp,
    int B, int T, int C, int NH) {

    int HS = C / NH; // head size

    // Permute and separate inp from (B, T, 3, NH, HS) to 3x (B, NH, T, HS)
    float* __restrict__ q = qkvr + 0 * B * T * C;
    float* __restrict__ k = qkvr + 1 * B * T * C;
    float* __restrict__ v = qkvr + 2 * B * T * C;
    
    dim3 permuteBlockDims(T);
    dim3 permuteGridDims(B, NH);
    permute_kernel<<<permuteGridDims, permuteBlockDims>>>(q, k, v, inp, B, T, NH, HS);
    cudaDeviceSynchronize();

    float* attention_out;
    cudaMalloc((void**)&attention_out, B * NH * T * HS * sizeof(float));

    dim3 num_threads(THREADS);
    dim3 num_blocks(CEIL_DIV(T, Q_TILE_SIZE), NH, B);
    float scale = 1.0f / sqrtf((float)HS);
    
    // Dynamic shared memory size calculation
    size_t smem_size = (Q_TILE_SIZE * HS + KV_TILE_SIZE * HS + KV_TILE_SIZE * HS) * sizeof(__half) +
                       (Q_TILE_SIZE * KV_TILE_SIZE + Q_TILE_SIZE * HS) * sizeof(float);
    
    flash_attn_kernel<<<num_blocks, num_threads, smem_size>>>(q, k, v, attention_out, B, NH, T, HS, scale);
    cudaDeviceSynchronize();

    dim3 unpermuteBlockDims(T);
    dim3 unpermuteGridDims(B, NH);
    unpermute_kernel<<<unpermuteGridDims, unpermuteBlockDims>>>(attention_out, out, B, T, NH, HS);
    cudaDeviceSynchronize();

    cudaFree(attention_out);
}

#endif // __ATTENTION_CUH__
