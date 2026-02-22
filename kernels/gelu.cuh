#ifndef GELU_KERNEL_CUH_
#define GELU_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include <math_constants.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / CUDART_PI_F)

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = inp[idx];
        float cube = 0.044715f * val * val * val;
        out[idx] = 0.5f * val * (1.0f + tanhf(GELU_SCALING_FACTOR * (val + cube)));
    }
}

// Launch kernel here
void gelu_forward(float* out, const float* inp, int N) {
    unsigned blockSize = 32;
    dim3 blockDims(blockSize);
    dim3 gridDims(CEIL_DIV(N, blockSize));

    gelu_forward_kernel<<<gridDims, blockDims>>>(out, inp, N);
}

#endif // GELU_KERNEL_CUH_
