#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include "../utils/cuda_utils.cuh"
#include "./matmul.cuh"
#include "./softmax.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// Launch all kernels related to attention here
void attention_forward(float *out, float *qkvr, float *att, float *inp, int B, int T, int C, int NH) {
    // Implement this

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;

    dim3 permuteBlockDims(T);
    dim3 permuteGridDims(B, NH);
    permute_kernel<<<permuteGridDims, permuteBlockDims>>>(q, k, v, inp, B, T, NH, HS);
    cudaDeviceSynchronize();

    // Attention matmul: Q @ K^T
    // Compute pre-attention scores (B, NH, T, T)
    float *preatt = inp;
    for (int i = 0; i < B * NH; i++) {
        const float *qptr = q + i * T * HS;
        const float *kptr = k + i * T * HS;
        float *preattptr = preatt + i * T * T;

        matmul_forward(preattptr, qptr, kptr, nullptr, 1, T, HS, T);
        cudaDeviceSynchronize();
    }

    // Compute the softmax
    float scale = 1.0 / sqrtf(HS);
    cudaMemset(att, 0, B * NH * T * T * sizeof(float));
    softmax_forward(att, scale, preatt, B * NH, T);
    cudaDeviceSynchronize();
    
    float *vaccum = inp;
    // Attention matmul: P @ V, where P holds the attention probabilities
    // (B, NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)

    // Create cuBLAS handle for P @ V computation
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < B * NH; i++) {
        const float *aptr = att + i * T * T;   // P: (T, T)
        const float *vptr = v + i * T * HS;    // V: (T, HS)
        float *vaccptr = vaccum + i * T * HS;  // out: (T, HS)

        // Compute: out = P @ V
        // P: (T, T), V: (T, HS), out: (T, HS)
        // In column-major: out^T = V^T @ P^T
        // So we compute: cublasSgemm with A=V, B=P, C=out
        // m = HS, n = T, k = T
        int m = HS;
        int n = T;
        int k = T;

        cublasSgemm(
            handle,
            CUBLAS_OP_N,    // V is not transposed
            CUBLAS_OP_N,    // P is not transposed
            m, n, k,        // dimensions
            &alpha,
            vptr, m,        // V matrix and leading dimension
            aptr, k,        // P matrix and leading dimension
            &beta,
            vaccptr, m      // out matrix and leading dimension
        );
    }

    cublasDestroy(handle);
    cudaDeviceSynchronize();

    dim3 unpermuteBlockDims(T);
    dim3 unpermuteGridDims(B, NH);
    unpermute_kernel<<<unpermuteGridDims, unpermuteBlockDims>>>(vaccum, out, B, T, NH, HS);
    cudaDeviceSynchronize();
}

#endif // __ATTENTION_CUH__
