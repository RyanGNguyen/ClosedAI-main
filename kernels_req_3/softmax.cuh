#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32

// Warp-level online softmax combine operation
// Takes two partial pairs (max_a, sum_a) and (max_b, sum_b) and merges them
static inline __device__ void online_combine(float &max_a, float &sum_a, float max_b, float sum_b) {
    float m_new = fmaxf(max_a, max_b);
    sum_a = sum_a * expf(max_a - m_new) + sum_b * expf(max_b - m_new);
    max_a = m_new;
}

static inline __device__ float2 warp_online_reduce(float maxval, float sumval) {
    unsigned mask = 0xffffffffu;
    for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float m_peer = __shfl_down_sync(mask, maxval, offset);
        float s_peer = __shfl_down_sync(mask, sumval, offset);
        online_combine(maxval, sumval, m_peer, s_peer);
    }
    return make_float2(maxval, sumval);
}

__global__ void softmax_forward_kernel(float *out, float inv_temperature, const float *inp, int N, int T) {
    int idx = blockIdx.x;
    if (idx >= N * T) return;

    int own_pos = idx % T;

    const float *x = inp + idx * T;
    float *y = out + idx * T;

    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    int lane = tid & (WARP_SIZE - 1);
    int warpId = tid / WARP_SIZE;
    int numWarps = (block_threads + WARP_SIZE - 1) / WARP_SIZE;

    extern __shared__ float shmem[];
    float *sh_max = shmem;
    float *sh_sum = shmem + numWarps;

    // Each thread processes strided elements [tid, tid + blockDim.x, ...] up to own_pos
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = tid; i <= own_pos; i += block_threads) {
        float val = inv_temperature * x[i];
        float m_new = fmaxf(local_max, val);
        local_sum = (local_sum == 0.0f) ? expf(val - m_new)
                                        : local_sum * expf(local_max - m_new) + expf(val - m_new);
        local_max = m_new;
    }

    // Warp-level reduction (online merge)
    float2 warp_res = warp_online_reduce(local_max, local_sum);
    float warp_max = warp_res.x;
    float warp_sum = warp_res.y;

    // Store one per warp
    if (lane == 0) {
        sh_max[warpId] = warp_max;
        sh_sum[warpId] = warp_sum;
    }
    __syncthreads();

    // Final block-wide reduction (first warp)
    float gmax = -FLT_MAX;
    float gsum = 0.0f;
    if (warpId == 0) {
        float m = (lane < numWarps) ? sh_max[lane] : -FLT_MAX;
        float s = (lane < numWarps) ? sh_sum[lane] : 0.0f;
        float2 g = warp_online_reduce(m, s);
        gmax = g.x;
        gsum = g.y;
        if (lane == 0) {
            sh_max[0] = gmax;
            sh_sum[0] = gsum;
        }
    }
    __syncthreads();

    gmax = sh_max[0];
    gsum = sh_sum[0];
    float inv_sum = 1.0f / gsum;

    // Final normalization and write outputs for [0..own_pos]
    for (int i = tid; i <= own_pos; i += block_threads) {
        float val = inv_temperature * x[i];
        float ev = expf(val - gmax);
        y[i] = ev * inv_sum;
    }
}

void softmax_forward(float *out, float inv_temperature, const float *inp, int N, int T) {
    int blockSize = 128;
    int numWarps = CEIL_DIV(blockSize, WARP_SIZE);
    size_t shmem_size = 2 * numWarps * sizeof(float);

    dim3 gridDims(N * T);
    dim3 blockDims(blockSize);

    softmax_forward_kernel<<<gridDims, blockDims, shmem_size>>>(out, inv_temperature, inp, N, T);
}

#endif // __SOFTMAX_KERNEL_CUH__
