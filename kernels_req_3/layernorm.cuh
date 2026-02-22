#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32

static inline __device__ float warp_sum(float val) {
    float partialSum = val;
    for (unsigned int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
    }
    return partialSum;
}

__global__ void layernorm_forward_kernel(float *out, float *mean, float *rstd, const float *inp, const float *weight,
                                         const float *bias, int B, int T, int C) {

    int b = blockIdx.y;
    int t = blockIdx.x;
    int token_start = (b * T + t) * C;

    // Accumulate local sums with memory coalescing
    float lmean = 0.0f;
    float lvar = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = inp[token_start + c];
        lmean += val;
        lvar += val * val;
    }
    
    // Reduce across all warps
    float w_mean = warp_sum(lmean);
    float w_var = warp_sum(lvar);

    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int numWarps = CEIL_DIV(blockDim.x, WARP_SIZE);

    extern __shared__ float buf[];
    float *buf_mean = buf;
    float *buf_var = buf + numWarps;

    // Write warp sums to shared memory
    if (lane == 0) {
        buf_mean[warpId] = w_mean;
        buf_var[warpId] = w_var;
    }
    __syncthreads();

    // Final reduction within first warp
    float g_mean = 0.0f;
    float g_rstd = 0.0f;
    if (warpId == 0) {
        // Load lane elments
        float s = (lane < numWarps) ? buf_mean[lane] : 0.0f;
        float ss = (lane < numWarps) ? buf_var[lane] : 0.0f;

        // Reduction
        s = warp_sum(s);
        ss = warp_sum(ss);

        // Compute token-wide mean and variance
        if (lane == 0) {
            g_mean = s / C;
            g_rstd = rsqrtf(ss / C - g_mean * g_mean + 1e-5f);
            
            mean[b*T + t] = g_mean;
            rstd[b*T + t] = g_rstd;
        }
    }
    __syncthreads();
    float m = mean[b*T + t];
    float r = rstd[b*T + t];
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
    // 1024 = 32 * WARP_SIZE
    int numWarps = min(CEIL_DIV(C, WARP_SIZE), 32);
    int blockSize = numWarps * WARP_SIZE;
    size_t shmem_size = 2 * numWarps * sizeof(float);

    layernorm_forward_kernel<<<gridDims, blockSize, shmem_size>>>(out, mean, rstd, inp, weight, bias, B, T, C);
}

#endif // __LAYERNORM_KERNEL_CUH__
