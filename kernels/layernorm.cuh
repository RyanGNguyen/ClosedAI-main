#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void layernorm_forward_kernel(float *out, float *mean, float *rstd, const float *inp, const float *weight,
                                         const float *bias, int B, int T, int C) {

    int b = blockIdx.y;
    int t = blockIdx.x;
    int token_start = (b * T + t) * C;

    extern __shared__ float buf[];
    float *buf_mean = buf;
    float *buf_var = buf + blockDim.x;

    // Accumulate partial row mean and variance using memory coalescing
    float lmean = 0.0f;
    float lvar = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = inp[token_start + c];
        lmean += val;
        lvar += val * val;
    }
    buf_mean[threadIdx.x] = lmean;
    buf_var[threadIdx.x] = lvar;
    __syncthreads();

    // Per-token logarithmic reduction into global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            buf_mean[threadIdx.x] += buf_mean[threadIdx.x + stride];
            buf_var[threadIdx.x] += buf_var[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute token-wide mean and variance
    float m = buf_mean[0] / C;
    float r = rsqrtf(buf_var[0] / C - m * m + 1e-5f);

    // Fused kernel optimization (Later)
    // if (threadIdx.x == 0) {
    //     mean[b*T + t] = m;
    //     rstd[b*T + t] = r;
    // }

    __syncthreads();

    // Normalize and scale + shift
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = inp[token_start + c];
        float norm = (val - m) * r;
        out[token_start + c] = norm * weight[c] + bias[c];
    }
}

// Launch kernel here
void layernorm_forward(float *out, float *mean, float *rstd, float *inp, float *weight, float *bias,
                       int B, int T, int C) {
    dim3 gridDims(T, B);
    // Get next power of 2 closest to C in order for sum reduction to work
    int nextPowTwo = (int)powf(2.0f, ceilf(log2f((float)C)));
    int blockSize = min(1024, nextPowTwo);
    size_t shmem_size = 2 * blockSize * sizeof(float);

    layernorm_forward_kernel<<<gridDims, blockSize, shmem_size>>>(out, mean, rstd, inp, weight, bias, B, T, C);
}

#endif // __LAYERNORM_KERNEL_CUH__
