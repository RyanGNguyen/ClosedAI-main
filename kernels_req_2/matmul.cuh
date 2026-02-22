#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel to add bias to the result
__global__ void add_bias_kernel(float *out, const float *bias, int B, int T, int OC) {
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;

    if (bt < B * T && oc < OC) {
        out[bt * OC + oc] += bias[oc];
    }
}

// Launch kernel here
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {
    // We want to compute: out = inp @ weight^T + bias
    // inp: (B*T, C) in row-major
    // weight: (OC, C) in row-major
    // out: (B*T, OC) in row-major

    // cuBLAS uses column-major. Row-major arrays viewed as column-major:
    // inp[B*T][C] row-major -> inp^T[C][B*T] column-major (lda=C)
    // weight[OC][C] row-major -> weight^T[C][OC] column-major (lda=C)
    // out[B*T][OC] row-major -> out^T[OC][B*T] column-major (lda=OC)
    //
    // We want: out = inp @ weight^T (row-major)
    // Equivalently: out^T = (weight^T)^T @ inp^T = weight @ inp^T (column-major)

    const float alpha = 1.0f;
    const float beta = 0.0f;

    int m = OC;      // rows of result (column-major)
    int n = B * T;   // cols of result (column-major)
    int k = C;       // inner dimension

    // Compute: out^T = weight @ inp^T (in column-major)
    // Use global cublas_handle from cuda_utils.cuh
    cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,  // transpose weight^T to get weight
        CUBLAS_OP_N,  // use inp^T as-is
        m, n, k,      // dimensions
        &alpha,
        weight, k,    // A matrix (weight^T in memory, C×OC, lda=C)
        inp, k,       // B matrix (inp^T in memory, C×B*T, lda=C)
        &beta,
        out, m        // C matrix (out^T in memory, OC×B*T, lda=OC)
    );

    // Add bias if present
    if (bias != NULL) {
        dim3 blockDim(16, 16);
        dim3 gridDim((B * T + blockDim.x - 1) / blockDim.x,
                     (OC + blockDim.y - 1) / blockDim.y);
        add_bias_kernel<<<gridDim, blockDim>>>(out, bias, B, T, OC);
        cudaDeviceSynchronize();
    }
}

#endif // __MATMUL_KERNEL_CUH__
