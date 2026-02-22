#ifndef __ATTENTION_NO_TC_CUH__
#define __ATTENTION_NO_TC_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>

#define Q_TILE_SIZE 16
#define KV_TILE_SIZE 32
#define WARP_SIZE 32
#define THREADS 128

__global__ void permute_kernel(float *q, float *k, float *v, const float *inp, int B, int N, int NH, int d) {
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

__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH, int d) {
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
// Flash attention without tensor cores - standard float computation
__global__ void flash_attn_kernel(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
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

    // Initialize Q data and output accumulator
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
        // Load K and V for current tile into shared memory
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

        // Compute Q @ K^T with causal masking
        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {
            int kv_idx = qk_idx % KV_TILE_SIZE;
            int q_idx = qk_idx / KV_TILE_SIZE;
            int q_offset = q_tile_offset + q_idx;
            int kv_offset = tile * KV_TILE_SIZE + kv_idx;

            if (q_offset < T && kv_offset < T && kv_offset <= q_offset) {
                float qk = 0.0f;
                for (int d = 0; d < D; d++) {
                    qk += s_q[q_idx * D + d] * s_K[kv_idx * D + d];
                }
                s_qK[q_idx * KV_TILE_SIZE + kv_idx] = qk * softmax_scale;
            } else {
                s_qK[q_idx * KV_TILE_SIZE + kv_idx] = -INFINITY;
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

        // Compute P @ V
        for (int qd_idx = tid; qd_idx < Q_TILE_SIZE * D; qd_idx += blockDim.x) {
            int q_idx = qd_idx / D;
            int d_idx = qd_idx % D;

            float pv = 0.0f;
            for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {
                pv += s_qK[q_idx * KV_TILE_SIZE + kv_idx] * s_V[kv_idx * D + d_idx];
            }
            s_o[qd_idx] += pv;
        }
        __syncthreads();

        // Compute and update sum_exp
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

// Launch all kernels related to attention here
void attention_forward_no_tc(float *out, float *qkvr, float *att, float *inp, int B, int T, int C, int NH) {
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
    
    flash_attn_kernel<<<num_blocks, num_threads, smem_size>>>(q, k, v, attention_out, B, NH, T, HS, scale);
    cudaDeviceSynchronize();

    // Fix: Use fixed number of threads, not T
    int unpermute_threads = min(256, T);
    dim3 unpermuteBlockDims(unpermute_threads);
    dim3 unpermuteGridDims(B, NH);
    unpermute_kernel<<<unpermuteGridDims, unpermuteBlockDims>>>(attention_out, out, B, T, NH, HS);
    cudaDeviceSynchronize();

    cudaFree(attention_out);
}

#endif // __ATTENTION_NO_TC_CUH__
