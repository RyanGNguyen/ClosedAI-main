#ifndef RESIDUAL_KERNEL_CUH_
#define RESIDUAL_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

// Launch kernel here
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    // Setting blockSize to # of warps for now
    unsigned blockSize = 32;
    dim3 blockDims(blockSize);
    dim3 gridDims(CEIL_DIV(N, blockSize));

    residual_forward_kernel<<<gridDims, blockDims>>>(out, inp1, inp2, N);
}

#endif // RESIDUAL_KERNEL_CUH_