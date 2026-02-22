#ifndef ENCODER_FORWARD_KERNEL_CUH
#define ENCODER_FORWARD_KERNEL_CUH

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"


__global__ void encoder_forward_kernel(
    float* __restrict__ out,       
    const int* __restrict__ inp,   
    const float* __restrict__ wte, 
    const float* __restrict__ wpe, 
    int B, int T, int C) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;

    if (idx < total) {
        int channel = idx % C;
        int token_pos = (idx / C) % T;
        int batch = idx / (T * C);

        int token = inp[batch * T + token_pos];
        out[idx] = wte[token * C + channel] + wpe[token_pos * C + channel];
    }
}

// Launch kernel here
void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    unsigned blockSize = 256;
    int total = B * T * C;
    unsigned gridSize = CEIL_DIV(total, blockSize);

    encoder_forward_kernel<<<gridSize, blockSize>>>(out, inp, wte, wpe, B, T, C);
}

#endif // ENCODER_FORWARD_KERNEL_CUH
