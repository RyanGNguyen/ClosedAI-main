#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Tensor Core tile dimensions for TF32: 16x16x8 (M x N x K)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define WARP_SIZE 32

// Kernel using Tensor Cores with TF32 precision
// Computes: out = inp @ weight^T + bias
// inp: (B*T, C), weight: (OC, C), out: (B*T, OC)
__global__ void matmul_tensor_core_kernel(float *out, const float *inp, const float *weight,
                                          const float *bias, int C, int OC, int B, int T) {
    // Each warp computes one WMMA_M x WMMA_N output tile
    int warpId = (threadIdx.x / WARP_SIZE);
    int laneId = threadIdx.x % WARP_SIZE;

    // Calculate which output tile this warp is responsible for
    int warpRow = blockIdx.y * (blockDim.x / WARP_SIZE) + warpId;
    int warpCol = blockIdx.x;

    int bt = warpRow * WMMA_M;  // Starting row in output (batch*token)
    int oc = warpCol * WMMA_N;  // Starting column in output (output channel)

    // Bounds check for this tile
    if (bt >= B * T || oc >= OC) return;

    // Declare fragments for Tensor Core operations
    // A: inp, B: weight^T (loaded as col_major), C: accumulator
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in chunks of WMMA_K
    for (int k = 0; k < C; k += WMMA_K) {
        if (bt + WMMA_M <= B * T && oc + WMMA_N <= OC && k + WMMA_K <= C) {
            // Load matrix A (input): WMMA_M x WMMA_K tile from inp
            wmma::load_matrix_sync(a_frag, inp + bt * C + k, C);

            // Load matrix B (weight^T): WMMA_K x WMMA_N tile
            // weight is (OC, C) row-major, load as col_major to get transpose
            wmma::load_matrix_sync(b_frag, weight + oc * C + k, C);

            // Perform Tensor Core matrix multiplication: acc += A @ B
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store the result
    if (bt + WMMA_M <= B * T && oc + WMMA_N <= OC) {
        wmma::store_matrix_sync(out + bt * OC + oc, acc_frag, OC, wmma::mem_row_major);
    }

    // Add bias if present (per-thread operation)
    if (bias != NULL && laneId == 0) {
        for (int i = 0; i < WMMA_M && (bt + i) < B * T; i++) {
            for (int j = 0; j < WMMA_N && (oc + j) < OC; j++) {
                out[(bt + i) * OC + (oc + j)] += bias[oc + j];
            }
        }
    }
}

// Launch kernel here
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {

    // Use 128 threads per block (4 warps)
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE; // 4 warps

    // Grid dimensions
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(
        (OC + WMMA_N - 1) / WMMA_N,           // Number of WMMA_N tiles in columns
        ((B * T + WMMA_M - 1) / WMMA_M + warpsPerBlock - 1) / warpsPerBlock  // Number of warp-blocks in rows
    );

    matmul_tensor_core_kernel<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC, B, T);
}

#endif // __MATMUL_KERNEL_CUH__
