//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Include CPU kernels
#include "cpu_kernels/attention.cuh"
#include "cpu_kernels/encoder.cuh"
#include "cpu_kernels/gelu.cuh"
#include "cpu_kernels/layernorm.cuh"
#include "cpu_kernels/matmul.cuh"
#include "cpu_kernels/residual.cuh"

// Include GPU kernels
#include "kernels/attention.cuh"
#include "kernels/encoder.cuh"
#include "kernels/gelu.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/matmul.cuh"
#include "kernels/residual.cuh"
#include "kernels/softmax.cuh"

// Include CUDA utilities
#include "utils/cuda_utils.cuh"

int main() {

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    const int B = 1; // Batch size
    const int T = 1024; // Sequence length
    const int C = 768; // Input channels
    const int OC = 4 * C; // Output channels

    const int NH = 12; // Number of heads

    const float error_threshold = 0.01; // Error threshold for GPU vs CPU output comparisons
                                        // This is in unit of percentage difference in output values

    const float ep = 0.0001; // Epsilon value for percentage calculation to avoid getting infinity

    // used in multi-head self-attention
    const int HS = C / NH; // head size

    // Generate random input data
    srand(time(NULL));

    /*------------------------ encoder_forward test ------------------------*/

    // Allocate memory for input data
    int* enc_inp_cpu = (int*)malloc(B * T * sizeof(int));
    float* wte_enc = (float*)malloc(C * C * sizeof(float));
    float* wpe_enc = (float*)malloc(T * C * sizeof(float));

    for (int i = 0; i < B * T; ++i) {
        enc_inp_cpu[i] = (int)rand() / RAND_MAX;
    }
    for (int i = 0; i < C * C; ++i) {
        wte_enc[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < T * C; ++i) {
        wpe_enc[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    int* enc_inp_gpu_d;
    float* wte_enc_d;
    float* wpe_enc_d;
    cudaMalloc(&enc_inp_gpu_d, B * T * sizeof(int));
    cudaMalloc(&wte_enc_d, C * C * sizeof(float));
    cudaMalloc(&wpe_enc_d, T * C * sizeof(float));
    cudaMemcpy(enc_inp_gpu_d, enc_inp_cpu, B * T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(wte_enc_d, wte_enc, C * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wpe_enc_d, wpe_enc, T * C * sizeof(float), cudaMemcpyHostToDevice);


    // Allocate memory for output
    float* enc_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* enc_out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* enc_out_gpu_d;
    cudaMalloc(&enc_out_gpu_d, B * T * C * sizeof(float));

    // Call CPU function for reference output
    encoder_forward_cpu(enc_out_cpu, enc_inp_cpu, wte_enc, wpe_enc, B, T, C);

    // Call GPU function
    encoder_forward(enc_out_gpu_d, enc_inp_gpu_d, wte_enc_d, wpe_enc_d, B, T, C);

    // Copy output from GPU function back to CPU
    cudaMemcpy(enc_out_gpu, enc_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int enc_totElements = B * T * C;
    int enc_error = 0;
    for (int i = 0; i < enc_totElements; i++) {
        if (fabs(enc_out_cpu[i] - enc_out_gpu[i]) / (enc_out_cpu[i] + ep) > error_threshold) {
            enc_error++;
        }
    }

    if (enc_error > 0) {
        printf("+++++++++++++++ encoder_forward test failed with %d differing values +++++++++++++++\n", enc_error);
        printf("+++++++++++++++  Check your implementation or error_threshold value  +++++++++++++++\n");
    } else {
        printf("--------------------encoder_forward test passed!--------------------\n");
    }

    /*------------------------ layernorm_forward test ------------------------*/

    // Allocate memory for input data
    float* lay_inp_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* weight_ln = (float*)malloc(C * sizeof(float));
    float* bias_ln = (float*)malloc(C * sizeof(float));
    float* rstd_ln = (float*)malloc(B * T * sizeof(float));
    float* mean_ln = (float*)malloc(B * T * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        lay_inp_cpu[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < B * T; ++i) {
        rstd_ln[i] = (float)rand() / RAND_MAX;
        mean_ln[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < C; ++i) {
        weight_ln[i] = (float)rand() / RAND_MAX;
        bias_ln[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* lay_inp_gpu_d;
    float* weight_ln_d;
    float* bias_ln_d;
    float* rstd_ln_d;
    float* mean_ln_d;
    cudaMalloc(&lay_inp_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&weight_ln_d, C * sizeof(float));
    cudaMalloc(&bias_ln_d, C * sizeof(float));
    cudaMalloc(&rstd_ln_d, B * T * sizeof(float));
    cudaMalloc(&mean_ln_d, B * T * sizeof(float));
    cudaMemcpy(lay_inp_gpu_d, lay_inp_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_ln_d, weight_ln, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_ln_d, bias_ln, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rstd_ln_d, rstd_ln, B * T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mean_ln_d, mean_ln, B * T * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for output
    float* lay_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* lay_out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* lay_out_gpu_d;
    cudaMalloc(&lay_out_gpu_d, B * T * C * sizeof(float));

    // Call CPU function for reference output
    layernorm_forward_cpu(lay_out_cpu, mean_ln, rstd_ln, lay_inp_cpu, weight_ln, bias_ln, B, T, C);

    // Call GPU function
    layernorm_forward(lay_out_gpu_d, mean_ln_d, rstd_ln_d, lay_inp_gpu_d, weight_ln_d, bias_ln_d, B, T, C);

    // Copy output from GPU function back to CPU
    cudaMemcpy(lay_out_gpu, lay_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int lay_totElements = B * T * C;
    int lay_error = 0;
    for (int i = 0; i < lay_totElements; i++) {
        if (fabs(lay_out_cpu[i] - lay_out_gpu[i])/(lay_out_cpu[i] + ep) > error_threshold) {
            lay_error++;
        }
    }

    if (lay_error > 0) {
        printf("+++++++++++++++ layernorm_forward test failed with %d differing values +++++++++++++++\n", lay_error);
        printf("+++++++++++++++  Check your implementation or error_threshold value  +++++++++++++++\n");
    } else {
        printf("--------------------layernorm_forward test passed!--------------------\n");
    }

    /*------------------------ softmax_forward test ------------------------*/

    // Allocate memory for input data
    const int N = B * NH; // Number of rows for softmax
    float* softmax_inp_cpu = (float*)malloc(N * T * T * sizeof(float));
    float* softmax_out_cpu = (float*)malloc(N * T * T * sizeof(float));
    float* softmax_out_gpu = (float*)malloc(N * T * T * sizeof(float));
    // Create separate copy of input for GPU
    float* softmax_inp_gpu = (float*)malloc(N * T * T * sizeof(float));

    // Generate random input data
    for (int i = 0; i < N * T * T; ++i) {
        softmax_inp_cpu[i] = (float)rand() / RAND_MAX;
        softmax_inp_gpu[i] = softmax_inp_cpu[i];
    }

    // Copy data to GPU
    float* softmax_inp_gpu_d;
    float* softmax_out_gpu_d;
    cudaMalloc(&softmax_inp_gpu_d, N * T * T * sizeof(float));
    cudaMalloc(&softmax_out_gpu_d, N * T * T * sizeof(float));
    cudaMemcpy(softmax_inp_gpu_d, softmax_inp_gpu, N * T * T * sizeof(float), cudaMemcpyHostToDevice);

    // Call CPU function for reference output
    float inv_temp = 1.0f / sqrtf(HS);
    softmax_forward_cpu(softmax_out_cpu, inv_temp, softmax_inp_cpu, N, T);

    // Call GPU function
    softmax_forward(softmax_out_gpu_d, inv_temp, softmax_inp_gpu_d, N, T);
    cudaDeviceSynchronize();

    // Copy output from GPU function back to CPU
    cudaMemcpy(softmax_out_gpu, softmax_out_gpu_d, N * T * T * sizeof(float), cudaMemcpyDeviceToHost);

    int softmax_totElements = N * T * T;
    int softmax_diff = 0;
    for (int i = 0; i < softmax_totElements; ++i) {
        if (fabs(softmax_out_cpu[i] - softmax_out_gpu[i]) / (softmax_out_cpu[i] + ep) > error_threshold) {
            ++softmax_diff;
        }
    }

    if (softmax_diff > 0) {
        printf("+++++++++++++++ softmax_forward test failed with %d differing values +++++++++++++++\n", softmax_diff);
        printf("+++++++++++++++ Check your implementation or error_threshold value  +++++++++++++++\n");
    } else {
        printf("--------------------softmax_forward test passed!--------------------\n");
    }

    /*------------------------ attention_forward test ------------------------*/

    // Allocate memory for input data
    int inp_size = B * T * max(3 * C, NH * T);
    float* att_inp_cpu = (float*)malloc(inp_size * sizeof(float));
    float* qkvr = (float*)malloc(3 * B * T * C * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    // create separate copy of input for GPU to avoid input memory being overwritten
    float* att_inp_gpu = (float*)malloc(inp_size * sizeof(float));

    // Generate random input data
    for (int i = 0; i < inp_size; ++i) {
        att_inp_cpu[i] = (float)rand() / RAND_MAX;
        att_inp_gpu[i] = att_inp_cpu[i];
    }

    // copy data to GPU
    float* att_inp_gpu_d;
    float* qkvr_d;
    float* att_d;
    cudaMalloc(&att_inp_gpu_d, inp_size * sizeof(float));
    cudaMalloc(&qkvr_d, 3 * B * T * C * sizeof(float));
    cudaMalloc(&att_d, B * NH * T * T * sizeof(float));
    cudaMemcpy(att_inp_gpu_d, att_inp_gpu, inp_size * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for output
    float* att_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* att_out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* att_out_gpu_d;
    cudaMalloc(&att_out_gpu_d, B * T * C * sizeof(float));

    // Call CPU function for reference output
    attention_forward_cpu(att_out_cpu, qkvr, att, att_inp_cpu, B, T, C, NH);

    // Call GPU function
    attention_forward(att_out_gpu_d, qkvr_d, att_d, att_inp_gpu_d, B, T, C, NH);

    // Copy output from GPU function back to CPU
    cudaMemcpy(att_out_gpu, att_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int att_totElements = B * T * C;
    int att_diff = 0;
    // Loop through the results to count differences
    for (int i = 0; i < att_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(att_out_cpu[i] - att_out_gpu[i])/(att_out_cpu[i] + ep) > error_threshold) {
            ++att_diff;
        }
    }

    if (att_diff > 0) {
        printf("+++++++++++++++ attention_forward test failed with %d differing values +++++++++++++++\n", att_diff);
        printf("+++++++++++++++   Check your implementation or error_threshold value   +++++++++++++++\n");
        printf("++++++++ Try testing each attention component separately for easier debugging ++++++++\n");
    } else {
        printf("--------------------attention_forward test passed!--------------------\n");
    }

    /*------------------------ residual_forward test ------------------------*/

    // Allocate memory for input data
    float* res_inp1_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* res_inp2_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* res_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* res_out_gpu = (float*)malloc(B * T * C * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        res_inp1_cpu[i] = (float)rand() / RAND_MAX;
        res_inp2_cpu[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* res_inp1_gpu_d;
    float* res_inp2_gpu_d;
    float* res_out_gpu_d;
    cudaMalloc(&res_inp1_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&res_inp2_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&res_out_gpu_d, B * T * C * sizeof(float));
    cudaMemcpy(res_inp1_gpu_d, res_inp1_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(res_inp2_gpu_d, res_inp2_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // Call CPU function for reference output
    residual_forward_cpu(res_out_cpu, res_inp1_cpu, res_inp2_cpu, B * T * C);

    // Call GPU function
    residual_forward(res_out_gpu_d, res_inp1_gpu_d, res_inp2_gpu_d, B * T * C);

    // Copy output from GPU function back to CPU
    cudaMemcpy(res_out_gpu, res_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int res_totElements = B * T * C;
    int res_diff = 0;
    // Loop through the results to count differences
    for (int i = 0; i < res_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(res_out_cpu[i] - res_out_gpu[i])/ (res_out_cpu[i] + ep) > error_threshold) {
            ++res_diff;
        }
    }

    if (res_diff > 0) {
        printf("+++++++++++++++ residual_forward test failed with %d differing values +++++++++++++++\n", res_diff);
        printf("+++++++++++++++  Check your implementation or error_threshold value   +++++++++++++++\n");
    } else {
        printf("--------------------residual_forward test passed!--------------------\n");
    }

    /*------------------------ gelu_forward test ------------------------*/

    // Allocate memory for input data
    float* gelu_inp_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* gelu_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* gelu_out_gpu = (float*)malloc(B * T * C * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        gelu_inp_cpu[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* gelu_inp_gpu_d;
    float* gelu_out_gpu_d;
    cudaMalloc(&gelu_inp_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&gelu_out_gpu_d, B * T * C * sizeof(float));
    cudaMemcpy(gelu_inp_gpu_d, gelu_inp_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // Call CPU function for reference output
    gelu_forward_cpu(gelu_out_cpu, gelu_inp_cpu, B * T * C);

    // Call GPU function
    gelu_forward(gelu_out_gpu_d, gelu_inp_gpu_d, B * T * C);

    // Copy output from GPU function back to CPU
    cudaMemcpy(gelu_out_gpu, gelu_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int gelu_totElements = B * T * C;
    int gelu_diff = 0;
    for (int i = 0; i < gelu_totElements; ++i) {
        if (fabs(gelu_out_cpu[i] - gelu_out_gpu[i])/(gelu_out_cpu[i] + ep) > error_threshold) {
            ++gelu_diff;
        }
    }

    if (gelu_diff > 0) {
        printf("+++++++++++++++ gelu_forward test failed with %d differing values +++++++++++++++\n", gelu_diff);
        printf("+++++++++++++++ Check your implementation or error_threshold value +++++++++++++++\n");
    } else {
        printf("--------------------gelu_forward test passed!--------------------\n");
    }

    /*------------------------ matmul_forward test ------------------------*/

    // Allocate memory for input data
    float* mat_inp_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* matmul_weight = (float*)malloc(4 * C * C * sizeof(float));
    float* matmul_bias = (float*)malloc(4 * C * sizeof(float));
    float* mat_out_cpu = (float*)malloc(B * T * 4 * C * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        mat_inp_cpu[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < 4 * C * C; ++i) {
        matmul_weight[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < 4 * C; ++i) {
        matmul_bias[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* mat_inp_gpu_d;
    float* matmul_weight_d;
    float* matmul_bias_d;
    float* mat_out_gpu_d;
    float* mat_out_gpu = (float*)malloc(B * T * 4 * C * sizeof(float));
    cudaMalloc(&mat_inp_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&matmul_weight_d, 4 * C * C * sizeof(float));
    cudaMalloc(&matmul_bias_d, 4 * C * sizeof(float));
    cudaMalloc(&mat_out_gpu_d, B * T * 4 * C * sizeof(float));
    cudaMemcpy(mat_inp_gpu_d, mat_inp_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matmul_weight_d, matmul_weight, 4 * C * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matmul_bias_d, matmul_bias, 4 * C * sizeof(float), cudaMemcpyHostToDevice);

    // Call CPU function for reference output
    matmul_forward_cpu(mat_out_cpu, mat_inp_cpu, matmul_weight, matmul_bias, B, T, C, OC);

    // Call GPU function
    matmul_forward(mat_out_gpu_d, mat_inp_gpu_d, matmul_weight_d, matmul_bias_d, B, T, C, OC);

    // Copy output from GPU function back to CPU
    cudaMemcpy(mat_out_gpu, mat_out_gpu_d, B * T * 4 * C * sizeof(float), cudaMemcpyDeviceToHost);

    int mat_totElements = B * T * 4 * C;
    int mat_diff = 0;
    for (int i = 0; i < mat_totElements; ++i) {
        if (fabs(mat_out_cpu[i] - mat_out_gpu[i])/(mat_out_cpu[i] + ep) > error_threshold) {
            ++mat_diff;
        }
    }

    if (mat_diff > 0) {
        printf("+++++++++++++++ matmul_forward test failed with %d differing values +++++++++++++++\n", mat_diff);
        printf("+++++++++++++++ Check your implementation or error_threshold value  +++++++++++++++\n");
    } else {
        printf("--------------------matmul_forward test passed!--------------------\n");
    }

    printf("********************* All Random Tests Finished! *********************\n");

    // Free allocated memory

    free(enc_inp_cpu);
    free(enc_out_cpu);
    free(wte_enc);
    free(wpe_enc);
    free(enc_out_gpu);
    cudaFree(enc_inp_gpu_d);
    cudaFree(wte_enc_d);
    cudaFree(wpe_enc_d);
    cudaFree(enc_out_gpu_d);

    free(lay_inp_cpu);
    free(lay_out_cpu);
    free(weight_ln);
    free(bias_ln);
    free(rstd_ln);
    free(mean_ln);
    free(lay_out_gpu);
    cudaFree(lay_inp_gpu_d);
    cudaFree(weight_ln_d);
    cudaFree(bias_ln_d);
    cudaFree(rstd_ln_d);
    cudaFree(mean_ln_d);
    cudaFree(lay_out_gpu_d);

    // free(perm_inp_cpu);
    // free(perm_q_cpu);
    // free(perm_k_cpu);
    // free(perm_v_cpu);
    // free(perm_q_gpu);
    // free(perm_k_gpu);
    // free(perm_v_gpu);
    // cudaFree(perm_inp_gpu_d);
    // cudaFree(perm_q_d);
    // cudaFree(perm_k_d);
    // cudaFree(perm_v_d);

    // free(unperm_inp_cpu);
    // free(unperm_out_cpu);
    // free(unperm_out_gpu);
    // cudaFree(unperm_inp_gpu_d);
    // cudaFree(unperm_out_gpu_d);

    free(softmax_inp_cpu);
    free(softmax_inp_gpu);
    free(softmax_out_cpu);
    free(softmax_out_gpu);
    cudaFree(softmax_inp_gpu_d);
    cudaFree(softmax_out_gpu_d);

    free(att_inp_cpu);
    free(att_out_cpu);
    free(att_inp_gpu);
    free(att_out_gpu);
    free(qkvr);
    free(att);
    cudaFree(att_inp_gpu_d);
    cudaFree(qkvr_d);
    cudaFree(att_d);
    cudaFree(att_out_gpu_d);

    free(res_inp1_cpu);
    free(res_inp2_cpu);
    free(res_out_cpu);
    free(res_out_gpu);
    cudaFree(res_inp1_gpu_d);
    cudaFree(res_inp2_gpu_d);
    cudaFree(res_out_gpu_d);

    free(gelu_inp_cpu);
    free(gelu_out_cpu);
    free(gelu_out_gpu);
    cudaFree(gelu_inp_gpu_d);
    cudaFree(gelu_out_gpu_d);

    free(mat_inp_cpu);
    free(mat_out_gpu);
    free(matmul_weight);
    free(matmul_bias);
    cudaFree(mat_inp_gpu_d);
    cudaFree(matmul_weight_d);
    cudaFree(matmul_bias_d);
    cudaFree(mat_out_gpu_d);

    return 0;
}
