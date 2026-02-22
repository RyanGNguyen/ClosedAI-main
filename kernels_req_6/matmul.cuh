#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_forward_kernel(float *out, const float *inp, const float *weight,
                                      const float *bias, int C, int OC, int B, int T) {
    // Implement this

    __shared__ float t_inp[TILE_WIDTH][TILE_WIDTH];
    __shared__ float t_weight[TILE_WIDTH][TILE_WIDTH];

    int bt = blockIdx.y * TILE_WIDTH + threadIdx.y; // token
    int oc = blockIdx.x * TILE_WIDTH + threadIdx.x; // output channel

    int num_tiles = (C + TILE_WIDTH - 1) / TILE_WIDTH;

    float acc = 0.0f;
    if (bias != NULL && oc < OC) acc = bias[oc];

    for (int tile = 0; tile < num_tiles; tile++) {
        int inp_col = tile * TILE_WIDTH + threadIdx.x;
        int weight_col = tile * TILE_WIDTH + threadIdx.y;

        if (bt < B * T && inp_col < C) {
            t_inp[threadIdx.y][threadIdx.x] = inp[bt * C + inp_col];
        } else {
            t_inp[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (oc < OC && weight_col < C) {
            t_weight[threadIdx.y][threadIdx.x] = weight[oc * C + weight_col];
        } else {
            t_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            acc += t_inp[threadIdx.y][k] * t_weight[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (bt < B * T && oc < OC) {
        out[bt * OC + oc] = acc;
    }

    // if (oc < OC) {
    //     // Initiate accumulator with bias

    //     const float *inp_token_ptr = inp + bt * C;     // Consider Matrix A (input Tokens * Batches)
    //     const float *weight_row_ptr = weight + oc * C; // Consider Matrix B (Weight)
    //                                                    // Basic matrix multiplication
    //     for (int i = 0; i < C; i++) {
    //         acc += inp_token_ptr[i] * weight_row_ptr[i];
    //     }

    //     out[bt * OC + oc] = acc;
    // }
}

// Launch kernel here
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {
    // Implement this

    dim3 blockDim(16, 16);

    dim3 gridDim((OC + blockDim.x - 1) / blockDim.x, (B * T + blockDim.y - 1) / blockDim.y);

    matmul_forward_kernel<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC, B, T);
}

#endif // __MATMUL_KERNEL_CUH__