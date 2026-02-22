#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include "../utils/cuda_utils.cuh"
#include "./matmul.cuh"
#include "./softmax.cuh"
#include <cuda_runtime.h>

__global__ void permute_kernel(float *q, float *k, float *v, const float *inp, int B, int N, int NH, int d) {
    int b = blockIdx.x;
    int nh_idx = blockIdx.y;

    // Use grid-stride loop to handle large N values
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

    // Use grid-stride loop to handle large N values
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        for (int i = 0; i < d; i++) {
            int inp_idx = (b * NH * N * d) + (nh_idx * N * d) + (t * d) + i;
            int out_idx = (b * N * NH * d) + (t * NH * d) + (nh_idx * d) + i;
            out[out_idx] = inp[inp_idx];
        }
    }
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

// Launch all kernels related to attention here
void attention_forward(float *out, float *qkvr, float *att, float *inp, int B, int T, int C, int NH) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;

    // Fix: Cap thread block size at 256 to avoid exceeding max threads (1024)
    int permute_threads = min(256, T);
    dim3 permuteBlockDims(permute_threads);
    dim3 permuteGridDims(B, NH);
    permute_kernel<<<permuteGridDims, permuteBlockDims>>>(q, k, v, inp, B, T, NH, HS);
    cudaDeviceSynchronize();

    // Attention matmul: Q @ K^T
    // Compute pre-attention scores (B, NH, T, T)
    // Use att buffer for preatt to avoid buffer overflow
    float *preatt = att;
    for (int i = 0; i < B * NH; i++) {
        const float *qptr = q + i * T * HS;
        const float *kptr = k + i * T * HS;
        float *preattptr = preatt + i * T * T;

        matmul_forward(preattptr, qptr, kptr, nullptr, 1, T, HS, T);
        cudaDeviceSynchronize();
    }

    // Compute the softmax
    float scale = 1.0 / sqrtf(HS);
    softmax_forward(att, scale, preatt, B * NH, T);
    cudaDeviceSynchronize();

    // Attention matmul: P @ V, where P holds the attention probabilities
    // (B, NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)
    // Use inp as temporary buffer for attention output (vaccum)
    float *vaccum = inp;
    for (int i = 0; i < B * NH; i++) {
        const float *aptr = att + i * T * T;
        const float *vptr = v + i * T * HS;
        float *vaccptr = vaccum + i * T * HS;

        dim3 blockDims(16, 16);
        dim3 gridDims(CEIL_DIV(T, 16), CEIL_DIV(HS, 16));
        pv_matmul_no_transpose_kernel<<<gridDims, blockDims>>>(vaccptr, aptr, vptr, T, HS);
    }
    cudaDeviceSynchronize();

    // Fix: Cap thread block size at 256 to avoid exceeding max threads (1024)
    int unpermute_threads = min(256, T);
    dim3 unpermuteBlockDims(unpermute_threads);
    dim3 unpermuteGridDims(B, NH);
    unpermute_kernel<<<unpermuteGridDims, unpermuteBlockDims>>>(vaccum, out, B, T, NH, HS);
    cudaDeviceSynchronize();
}

#endif // __ATTENTION_CUH__
