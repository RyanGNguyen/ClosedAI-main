#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>

// Shared memory tile size
#define TILE_M 64
#define TILE_N 64
#define TILE_S 8

// Register tile size 
#define TM 8
#define TN 8

// Thread block size
#define BM (TILE_M/TM) 
#define BN (TILE_N/TN)  

__global__ void matmul_forward_kernel(float *out, const float *inp, const float *weight,
                                      const float *bias, int C, int OC, int B, int T) {
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Global starting positions for output tile
    const int row_start = by * TILE_M;
    const int col_start = bx * TILE_N;
    
    // Shared memory
    __shared__ float As[TILE_M][TILE_S];
    __shared__ float Bs[TILE_S][TILE_N];
    
    // Register arrays for accumulation
    float acc[TM][TN];
    
    // Initialize
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            int row = row_start + ty * TM + i;
            int col = col_start + tx * TN + j;
            if(bias != nullptr && col < OC) {
                acc[i][j] = bias[col];
            } else {
                acc[i][j] = 0.0f;
            }
        }
    }
    
    const int num_tiles_k = (C + TILE_S - 1) / TILE_S;
    
    // Main loop over K tiles
    for(int k_tile = 0; k_tile < num_tiles_k; k_tile++) {
        #pragma unroll
        for(int m = 0; m < TILE_M; m += BM) {
            #pragma unroll
            for(int k = 0; k < TILE_S; k += BN) {
                int row = row_start + m + ty;
                int col = k_tile * TILE_S + k + tx;
                if(row < B * T && col < C) {
                    As[m + ty][k + tx] = inp[row * C + col];
                } else {
                    As[m + ty][k + tx] = 0.0f;
                }
            }
        }
        
        // Load tile from weight (OC x C)
        #pragma unroll
        for(int k = 0; k < TILE_S; k += BM) {
            #pragma unroll
            for(int n = 0; n < TILE_N; n += BN) {
                int row = col_start + n + tx;  // OC dimension
                int col = k_tile * TILE_S + k + ty;  // C dimension
                if(row < OC && col < C) {
                    Bs[k + ty][n + tx] = weight[row * C + col];
                } else {
                    Bs[k + ty][n + tx] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Register-level computation
        #pragma unroll
        for(int k = 0; k < TILE_S; k++) {
            // Load values into registers
            float a_reg[TM];
            float b_reg[TN];
            
            #pragma unroll
            for(int i = 0; i < TM; i++) {
                a_reg[i] = As[ty * TM + i][k];
            }
            
            #pragma unroll
            for(int j = 0; j < TN; j++) {
                b_reg[j] = Bs[k][tx * TN + j];
            }
            
            // Compute outer product
            #pragma unroll
            for(int i = 0; i < TM; i++) {
                #pragma unroll
                for(int j = 0; j < TN; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            int row = row_start + ty * TM + i;
            int col = col_start + tx * TN + j;
            if(row < B * T && col < OC) {
                out[row * OC + col] = acc[i][j];
            }
        }
    }
}

// Normal kernel for small matrices
__global__ void matmul_forward_kernel_small(float *out, const float *inp, const float *weight,
                                           const float *bias, int C, int OC, int B, int T) {
    #define TILE_WIDTH 16
    __shared__ float t_inp[TILE_WIDTH][TILE_WIDTH];
    __shared__ float t_weight[TILE_WIDTH][TILE_WIDTH];
    
    int bt = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int oc = blockIdx.x * TILE_WIDTH + threadIdx.x;
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
}

// Launch kernel here
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {
    if (B * T >= 64 && OC >= 64 && C >= 64) {
        // Use optimized kernel for larger matrices
        dim3 blockDim(BN, BM);
        dim3 gridDim((OC + TILE_N - 1) / TILE_N, (B * T + TILE_M - 1) / TILE_M);
        matmul_forward_kernel<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC, B, T);
    } else {
        // Use simple kernel for smaller matrices
        dim3 blockDim(16, 16);
        dim3 gridDim((OC + 15) / 16, (B * T + 15) / 16);
        matmul_forward_kernel_small<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC, B, T);
    }
}
#endif