#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_forward_kernel(float *out, float inv_temperature, const float *inp, int N, int T) {
    // Each thread processes one row
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N * T)
        return;

    // Position in the current row (for causal masking)
    int own_pos = idx % T;

    // Pointer to the current row
    const float *x = inp + idx * T;

    float maxval = -FLT_MAX;
    for (int i = 0; i <= own_pos; ++i) {
        maxval = fmaxf(maxval, x[i]);
    }

    // Compute exp and sum in one pass
    float sumval = 0.0f;
    for (int i = 0; i <= own_pos; ++i) {
        float ev = expf(inv_temperature * (x[i] - maxval));
        sumval += ev;
        out[idx * T + i] = ev;
    }

    // Normalize
    float norm = 1.0f / sumval;
    for (int i = 0; i <= own_pos; ++i) {
        out[idx * T + i] *= norm;
    }
}

// Launch kernel here
void softmax_forward(float *out, float inv_temperature, const float *inp, int N, int T) {
    unsigned blockSize = 32;
    dim3 blockDims(blockSize);
    dim3 gridDims(CEIL_DIV(N * T, blockSize));

    softmax_forward_kernel<<<gridDims, blockDims>>>(out, inv_temperature, inp, N, T);
}

#endif // __SOFTMAX_KERNEL_CUH__
