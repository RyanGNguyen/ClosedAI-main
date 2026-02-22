//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../utils/utils.h"
#include "../utils/cuda_utils.cuh"

#include "../gpt2.cuh"
#include "../kernels_awq/auto_scale/grid_search.cuh"

const char *channel_averages_filename = "awq_channel_avgs.bin";
const char *calibration_tokens_filename = "calibration_tokens.bin";
const char *optimal_alphas_filename = "awq_optimal_alphas.bin";
const char *optimal_scales_filename = "awq_optimal_scales.bin";
const char *scaled_params_filename = "awq_scaled_params.bin";
const char *checkpoint_file = "/content/drive/MyDrive/gpt2_124M.bin";
#define GRID_SEARCH_BATCH_SIZE 32  /* Must match calibration batch size */

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

int* load_calibration_tokens(int* num_batches_out, int T, int B, const char* input_file) {
    FILE* fp = fopenCheck(input_file, "rb");
    
    // Get file size to determine number of sequences
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    int num_sequences = file_size / (T * sizeof(int));
    int num_batches = num_sequences / B;
    
    printf("AWQ - Loading %d token sequences (%d batches) from %s\n", 
           num_sequences, num_batches, input_file);
    
    // Allocate and read directly into batched format (num_sequences * T)
    int* batched_tokens = (int*)mallocCheck(num_sequences * T * sizeof(int));
    size_t items_read = fread(batched_tokens, sizeof(int), num_sequences * T, fp);
    
    if (items_read != num_sequences * T) {
        fprintf(stderr, "Error: Expected to read %d tokens, got %zu\n", 
                num_sequences * T, items_read);
        exit(EXIT_FAILURE);
    }
    
    fcloseCheck(fp);
    
    *num_batches_out = num_batches;
    
    printf("AWQ - Loaded %d token sequences successfully\n", num_sequences);
    
    return batched_tokens;
}

void save_optimal_alphas(const float* h_alphas, int L, const char* output_file) {
    // h_alphas is (L, 4) - one alpha per (layer, matrix_type)
    FILE* fp = fopenCheck(output_file, "wb");
    
    // Write header
    fwrite(&L, sizeof(int), 1, fp);
    int num_matrices = 4;
    fwrite(&num_matrices, sizeof(int), 1, fp);
    
    // Write alphas: (L, 4)
    fwrite(h_alphas, sizeof(float), L * 4, fp);
    
    fcloseCheck(fp);
    
    printf("AWQ - Saved optimal alphas (%d x %d) to %s\n", L, 4, output_file);
}

void save_optimal_scales(const float* d_scales, int L, int C, const char* output_file) {
    // d_scales is (L*C*7) on device - same layout as channel averages
    size_t total_size = L * C * 7;
    
    float* h_scales = (float*)mallocCheck(total_size * sizeof(float));
    cudaCheck(cudaMemcpy(h_scales, d_scales, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* fp = fopenCheck(output_file, "wb");
    
    // Write header
    fwrite(&L, sizeof(int), 1, fp);
    fwrite(&C, sizeof(int), 1, fp);
    
    // Write scales
    fwrite(h_scales, sizeof(float), total_size, fp);
    
    fcloseCheck(fp);
    
    printf("AWQ - Saved optimal scales (%zu values) to %s\n", total_size, output_file);
    
    free(h_scales);
}

void save_scaled_params(GPT2* model, const char* output_file) {
    // After apply_optimal_scales_to_model(), save the scaled weights and biases
    int L = model->config.num_layers;
    int C = model->config.channels;
    
    printf("\n+----------------------------------+\n");
    printf("| SAVING SCALED PARAMETERS         |\n");
    printf("+----------------------------------+\n");
    
    FILE* fp = fopenCheck(output_file, "wb");
    
    // Write header
    fwrite(&L, sizeof(int), 1, fp);
    fwrite(&C, sizeof(int), 1, fp);
    
    // Calculate sizes for each parameter
    size_t qkvw_size = L * 3 * C * C;      // (L, 3*C, C)
    size_t attprojw_size = L * C * C;      // (L, C, C)
    size_t fcw_size = L * 4 * C * C;       // (L, 4*C, C)
    size_t fcprojw_size = L * C * 4 * C;   // (L, C, 4*C)
    size_t ln1w_size = L * C;              // (L, C)
    size_t ln1b_size = L * C;              // (L, C)
    size_t ln2w_size = L * C;              // (L, C)
    size_t ln2b_size = L * C;              // (L, C)
    size_t qkvb_size = L * 3 * C;          // (L, 3*C)
    size_t fcb_size = L * 4 * C;           // (L, 4*C)
    
    size_t total_size = qkvw_size + attprojw_size + fcw_size + fcprojw_size +
                        ln1w_size + ln1b_size + ln2w_size + ln2b_size +
                        qkvb_size + fcb_size;
    
    // Allocate host buffer
    float* h_params = (float*)mallocCheck(total_size * sizeof(float));
    
    // Copy all scaled parameters from device to host
    size_t offset = 0;
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.qkvw, qkvw_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += qkvw_size;
    printf("  Copied qkvw: %zu floats\n", qkvw_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.attprojw, attprojw_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += attprojw_size;
    printf("  Copied attprojw: %zu floats\n", attprojw_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.fcw, fcw_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += fcw_size;
    printf("  Copied fcw: %zu floats\n", fcw_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.fcprojw, fcprojw_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += fcprojw_size;
    printf("  Copied fcprojw: %zu floats\n", fcprojw_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.ln1w, ln1w_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += ln1w_size;
    printf("  Copied ln1w: %zu floats\n", ln1w_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.ln1b, ln1b_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += ln1b_size;
    printf("  Copied ln1b: %zu floats\n", ln1b_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.ln2w, ln2w_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += ln2w_size;
    printf("  Copied ln2w: %zu floats\n", ln2w_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.ln2b, ln2b_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += ln2b_size;
    printf("  Copied ln2b: %zu floats\n", ln2b_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.qkvb, qkvb_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += qkvb_size;
    printf("  Copied qkvb: %zu floats\n", qkvb_size);
    
    cudaCheck(cudaMemcpy(h_params + offset, model->params.fcb, fcb_size * sizeof(float), cudaMemcpyDeviceToHost));
    offset += fcb_size;
    printf("  Copied fcb: %zu floats\n", fcb_size);
    
    // Write all parameters to file
    fwrite(h_params, sizeof(float), total_size, fp);
    
    fcloseCheck(fp);
    free(h_params);
    
    printf("  Total: %zu floats (%zu MB)\n", total_size, (total_size * sizeof(float)) >> 20);
    printf("  Saved to: %s\n", output_file);
    printf("+----------------------------------+\n\n");
}

void run_auto_scaling_grid_search(GPT2* model, int* batched_tokens, int num_samples) {
    int L = model->config.num_layers;
    int C = model->config.channels;
    
    printf("\n+----------------------------------+\n");
    printf("| AWQ AUTO-SCALING GRID SEARCH     |\n");
    printf("+----------------------------------+\n");
    printf("Layers: %d, Channels: %d\n", L, C);
    printf("Num samples: %d\n", num_samples);
    printf("\n");
    
    // Load channel averages from calibration
    FILE* fp = fopenCheck(channel_averages_filename, "rb");
    int file_L, file_C;
    size_t file_tokens;
    freadCheck(&file_L, sizeof(int), 1, fp);
    freadCheck(&file_C, sizeof(int), 1, fp);
    freadCheck(&file_tokens, sizeof(size_t), 1, fp);
    
    if (file_L != L || file_C != C) {
        fprintf(stderr, "Error: Channel averages file dimensions don't match model\n");
        fprintf(stderr, "  File: L=%d, C=%d | Model: L=%d, C=%d\n", file_L, file_C, L, C);
        exit(EXIT_FAILURE);
    }
    
    size_t total_size = L * C * 7;
    float* h_channel_avgs = (float*)mallocCheck(total_size * sizeof(float));
    freadCheck(h_channel_avgs, sizeof(float), total_size, fp);
    fcloseCheck(fp);
    
    printf("AWQ - Loaded channel averages from %s\n", channel_averages_filename);
    
    // Copy to device
    float* d_channel_avgs;
    cudaCheck(cudaMalloc(&d_channel_avgs, total_size * sizeof(float)));
    cudaCheck(cudaMemcpy(d_channel_avgs, h_channel_avgs, total_size * sizeof(float), 
                        cudaMemcpyHostToDevice));
    free(h_channel_avgs);
    
    // Prepare alpha values for grid search
    float alpha_values[11] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    int num_alpha_values = 11;
    
    // Allocate output arrays
    float* h_optimal_alphas = (float*)mallocCheck(L * 4 * sizeof(float));  // (L, 4)
    float* d_optimal_scales;
    cudaCheck(cudaMalloc(&d_optimal_scales, total_size * sizeof(float)));  // (L*C*7)
    
    // Run grid search with batched tokens
    grid_search_optimal_alpha(
        h_optimal_alphas,
        d_optimal_scales,
        d_channel_avgs,
        model,
        batched_tokens,
        num_samples,
        num_alpha_values,
        alpha_values
    );
    
    // Save results
    save_optimal_alphas(h_optimal_alphas, L, optimal_alphas_filename);
    save_optimal_scales(d_optimal_scales, L, C, optimal_scales_filename);
    
    // Cleanup
    free(h_optimal_alphas);
    cudaCheck(cudaFree(d_channel_avgs));
    cudaCheck(cudaFree(d_optimal_scales));
    
    printf("\n+----------------------------------+\n");
    printf("| AUTO-SCALING COMPLETE            |\n");
    printf("+----------------------------------+\n\n");
}

void init_cublas() {
    cublasCheck(cublasCreate(&cublas_handle));
    int enable_tf32 = 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

void print_model_info(const GPT2* model) {
    printf("+-----------------------+\n");
    printf("| GPT2 MODEL PARAMETERS |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| max_sequence_length T | %-50d |\n", model->config.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model->config.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model->config.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model->config.num_layers);
    printf("| num_heads NH          | %-50d |\n", model->config.num_heads);
    printf("| channels C            | %-50d |\n", model->config.channels);
    printf("| num_parameters        | %-50zu |\n", model->num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");
}

int main(int argc, char** argv) {
    srand(time(NULL));
    init_cublas();

    GPT2 model;
    gpt2_build_from_checkpoint(&model, checkpoint_file);
    print_model_info(&model);

    int T = model.config.max_seq_len;
    int B = GRID_SEARCH_BATCH_SIZE;

    // Load calibration tokens (already in batched format)
    printf("\n>>> Loading calibration data...\n");
    int num_batches = 0;
    int* batched_tokens = load_calibration_tokens(&num_batches, T, B, calibration_tokens_filename);
    
    // Run auto-scaling grid search (applies optimal scales to model in-place)
    printf(">>> Starting AWQ auto-scaling grid search...\n");
    run_auto_scaling_grid_search(&model, batched_tokens, num_batches);
    
    // Save the scaled parameters (weights and biases have been modified by grid search)
    printf(">>> Saving scaled parameters...\n");
    save_scaled_params(&model, scaled_params_filename);
    
    // Cleanup
    free(batched_tokens);
    
    gpt2_free(&model);

    return EXIT_SUCCESS;
}
