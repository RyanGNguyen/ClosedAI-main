#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include "../utils/cuda_utils.cuh"
#include "./matmul.cuh"
#include "./softmax.cuh"
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Tensor Core tile dimensions for TF32: 16x16x8 (M x N x K)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define WARP_SIZE 32

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

// Tensor Core kernel for P @ V operation
// P: (T, T), V: (T, HS), out: (T, HS)
__global__ void pv_matmul_tensor_core_kernel(float *out, const float *P, const float *V, int T, int HS) {
    // Each warp computes one WMMA_M x WMMA_N output tile
    int warpId = (threadIdx.x / WARP_SIZE);

    // Calculate which output tile this warp is responsible for
    int warpRow = blockIdx.y * (blockDim.x / WARP_SIZE) + warpId;
    int warpCol = blockIdx.x;

    int row = warpRow * WMMA_M;  // Starting row in output
    int col = warpCol * WMMA_N;  // Starting column in output

    // Bounds check for this tile
    if (row >= T || col >= HS) return;

    // Declare fragments for Tensor Core operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in chunks of WMMA_K
    for (int k = 0; k < T; k += WMMA_K) {
        if (row + WMMA_M <= T && col + WMMA_N <= HS && k + WMMA_K <= T) {
            // Load matrix A (P): WMMA_M x WMMA_K tile
            wmma::load_matrix_sync(a_frag, P + row * T + k, T);

            // Load matrix B (V): WMMA_K x WMMA_N tile
            wmma::load_matrix_sync(b_frag, V + k * HS + col, HS);

            // Perform Tensor Core matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store the result
    if (row + WMMA_M <= T && col + WMMA_N <= HS) {
        wmma::store_matrix_sync(out + row * HS + col, acc_frag, HS, wmma::mem_row_major);
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

    // Use Tensor Cores for P @ V
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;

    for (int i = 0; i < B * NH; i++) {
        const float *aptr = att + i * T * T;
        const float *vptr = v + i * T * HS;
        float *vaccptr = vaccum + i * T * HS;

        dim3 blockDims(threadsPerBlock);
        dim3 gridDims(
            (HS + WMMA_N - 1) / WMMA_N,
            ((T + WMMA_M - 1) / WMMA_M + warpsPerBlock - 1) / warpsPerBlock
        );
        pv_matmul_tensor_core_kernel<<<gridDims, blockDims>>>(vaccptr, aptr, vptr, T, HS);
    }
    cudaDeviceSynchronize();

    dim3 unpermuteBlockDims(T);
    dim3 unpermuteGridDims(B, NH);
    unpermute_kernel<<<unpermuteGridDims, unpermuteBlockDims>>>(vaccum, out, B, T, NH, HS);
    cudaDeviceSynchronize();
}

#endif // __ATTENTION_CUH__
