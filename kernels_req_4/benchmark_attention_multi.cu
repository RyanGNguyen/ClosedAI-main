//-----------------------------------------------------------------------------------------------
// Attention Kernel Benchmark - Measures execution time without model dependencies
// Supports profiling three different attention kernel implementations
//-----------------------------------------------------------------------------------------------

#include "utils/utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <vector>

// Conditionally include the appropriate attention kernel based on preprocessor define
#ifdef PROFILE_FLASH_TC
    #include "kernels/attention.cuh"
    #define KERNEL_NAME "Flash Attention (Tensor Cores)"
#elif PROFILE_FLASH_NO_TC
    #include "kernels_req_4/attention_no_tc.cuh"
    #define KERNEL_NAME "Flash Attention (No Tensor Cores)"
#elif PROFILE_CLASSIC
    #include "kernels_req_3/attention.cuh"
    #define KERNEL_NAME "Classic Attention (cuBLAS)"
#else
    // Default to Flash Attention with Tensor Cores
    #include "kernels/attention.cuh"
    #define KERNEL_NAME "Flash Attention (Tensor Cores) [DEFAULT]"
#endif

// Configuration parameters for attention kernel
struct BenchmarkConfig {
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    int q_tile_size;
    int kv_tile_size;
    int attn_threads;
};

// cuBLAS handle for matmul operations
extern cublasHandle_t cublas_handle;
cudaStream_t stream;

void benchmark_attention(const BenchmarkConfig& config) {
    printf("\n=== Attention Benchmark ===\n");
    printf("Kernel: %s\n", KERNEL_NAME);
    printf("Batch: %d, Seq: %d, Heads: %d, HeadDim: %d\n",
           config.batch_size, config.seq_len, config.num_heads, config.head_dim);
    printf("Q_Tile: %d, KV_Tile: %d, Threads: %d\n",
           config.q_tile_size, config.kv_tile_size, config.attn_threads);

    int B = config.batch_size;
    int T = config.seq_len;
    int NH = config.num_heads;
    int d = config.head_dim;
    int C = NH * d;  // Total channel dimension

    // Allocate input tensors
    float *qkvr, *att, *output;
    size_t qkv_size = B * T * 3 * C * sizeof(float);
    size_t att_size = B * NH * T * T * sizeof(float);
    size_t output_size = B * T * C * sizeof(float);

    cudaCheck(cudaMalloc(&qkvr, qkv_size));
    cudaCheck(cudaMalloc(&att, att_size));
    cudaCheck(cudaMalloc(&output, output_size));

    // Warmup
#ifdef PROFILE_FLASH_NO_TC
    attention_forward_no_tc(output, qkvr, att, qkvr, B, T, C, NH);
#else
    attention_forward(output, qkvr, att, qkvr, B, T, C, NH);
#endif
    cudaCheck(cudaDeviceSynchronize());

    // Benchmark loop
    int num_runs = 10;
    float total_time = 0.0f;
    float min_time = FLT_MAX;
    float max_time = 0.0f;

    for (int run = 0; run < num_runs; run++) {
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));

        cudaCheck(cudaEventRecord(start, stream));
#ifdef PROFILE_FLASH_NO_TC
        attention_forward_no_tc(output, qkvr, att, qkvr, B, T, C, NH);
#else
        attention_forward(output, qkvr, att, qkvr, B, T, C, NH);
#endif
        cudaCheck(cudaEventRecord(stop, stream));
        cudaCheck(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));

        total_time += milliseconds;
        min_time = fminf(min_time, milliseconds);
        max_time = fmaxf(max_time, milliseconds);

        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
    }

    float avg_time = total_time / num_runs;

    // Calculate throughput (FLOPs)
    // Attention: Q@K^T (BTxD x DxT = BTxT) + softmax + V@output
    // Approximate FLOPs: 2 * B * T * T * d + 5 * B * NH * T * T (for softmax) + 2 * B * T * T * d
    long long flops = (long long)(2 * B * T * T * d + 5 * B * NH * T * T + 2 * B * T * T * d) * NH;
    double throughput = (flops / 1e9) / (avg_time / 1000.0);  // GFLOPS

    // Calculate total memory
    size_t total_memory = qkv_size + att_size + output_size;

    printf("Time (ms): avg=%.4f, min=%.4f, max=%.4f\n", avg_time, min_time, max_time);
    printf("Throughput: %.2f GFLOPS\n", throughput);
    printf("Memory Used: QKV=%.1f MB, Att=%.1f MB, Out=%.1f MB\n",
           qkv_size / 1e6, att_size / 1e6, output_size / 1e6);
    printf("Total Kernel Memory: %.1f MB\n", total_memory / 1e6);

    // Cleanup
    cudaCheck(cudaFree(qkvr));
    cudaCheck(cudaFree(att));
    cudaCheck(cudaFree(output));
}

int main(int argc, char *argv[]) {
    printf("=== Attention Kernel Benchmark ===\n");
    printf("Profiling: %s\n", KERNEL_NAME);

    // Set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device: %s\n", deviceProp.name);

    // Setup cuBLAS
    cublasCheck(cublasCreate(&cublas_handle));
    cudaCheck(cudaStreamCreate(&stream));

    // Define test configurations
    std::vector<BenchmarkConfig> configs = {
        // Standard attention config: batch=4, seq=1024, heads=12, head_dim=64
        {4, 4096, 2, 64 , 16, 64, 128},
    };

    // Run benchmarks
    for (const auto& config : configs) {
        try {
            benchmark_attention(config);
        } catch (const std::exception& e) {
            printf("Benchmark failed: %s\n", e.what());
        }
    }

    // Cleanup
    cublasCheck(cublasDestroy(cublas_handle));
    cudaCheck(cudaStreamDestroy(stream));

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
