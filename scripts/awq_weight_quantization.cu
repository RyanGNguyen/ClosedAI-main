//-----------------------------------------------------------------------------------------------
// Weight Quantization for AWQ (Activation-aware Weight Quantization)
// Quantizes GPT2 weights to 4-bit with per-group scaling and salient channel boosting
//
// Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita
// ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../utils/utils.h"
#include "../utils/cuda_utils.cuh"

#include "../gpt2.cuh"
#include "../kernels_awq/quantize.cuh"
#include "../kernels_awq/scale_tensors.cuh"

const char *checkpoint_file = "/content/drive/MyDrive/gpt2_124M.bin";
const char *scaled_params_filename = "awq_scaled_params.bin";
const char *output_checkpoint_file = "gpt2_124M_awq.bin";

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Structure to hold pre-scaled parameters from grid search
typedef struct {
    float* qkvw;         // (L, 3*C, C) - device pointer
    float* attprojw;     // (L, C, C) - device pointer
    float* fcw;          // (L, 4*C, C) - device pointer
    float* fcprojw;      // (L, C, 4*C) - device pointer
    float* ln1w;         // (L, C) - device pointer
    float* ln1b;         // (L, C) - device pointer
    float* ln2w;         // (L, C) - device pointer
    float* ln2b;         // (L, C) - device pointer
    float* qkvb;         // (L, 3*C) - device pointer
    float* fcb;          // (L, 4*C) - device pointer
} ScaledParams;

void load_scaled_params(const char* filename, ScaledParams* params, int L, int C) {
    printf("\n+----------------------------------+\n");
    printf("| LOADING SCALED PARAMETERS        |\n");
    printf("+----------------------------------+\n");
    
    FILE* fp = fopenCheck(filename, "rb");
    
    // Read header
    int file_L, file_C;
    freadCheck(&file_L, sizeof(int), 1, fp);
    freadCheck(&file_C, sizeof(int), 1, fp);
    
    if (file_L != L || file_C != C) {
        fprintf(stderr, "Error: Scaled params file dimensions don't match model\n");
        fprintf(stderr, "  File: L=%d, C=%d | Model: L=%d, C=%d\n", file_L, file_C, L, C);
        exit(EXIT_FAILURE);
    }
    
    // Calculate sizes for each parameter
    size_t qkvw_size = L * 3 * C * C;
    size_t attprojw_size = L * C * C;
    size_t fcw_size = L * 4 * C * C;
    size_t fcprojw_size = L * C * 4 * C;
    size_t ln1w_size = L * C;
    size_t ln1b_size = L * C;
    size_t ln2w_size = L * C;
    size_t ln2b_size = L * C;
    size_t qkvb_size = L * 3 * C;
    size_t fcb_size = L * 4 * C;
    
    size_t total_size = qkvw_size + attprojw_size + fcw_size + fcprojw_size +
                        ln1w_size + ln1b_size + ln2w_size + ln2b_size +
                        qkvb_size + fcb_size;
    
    // Read all parameters from file
    float* h_params = (float*)mallocCheck(total_size * sizeof(float));
    freadCheck(h_params, sizeof(float), total_size, fp);
    fcloseCheck(fp);
    
    printf("  Loaded %zu floats (%zu MB) from %s\n", 
           total_size, (total_size * sizeof(float)) >> 20, filename);
    
    // Allocate device memory and copy each parameter
    size_t offset = 0;
    
    cudaCheck(cudaMalloc(&params->qkvw, qkvw_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->qkvw, h_params + offset, qkvw_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += qkvw_size;
    printf("  Copied qkvw to device\n");
    
    cudaCheck(cudaMalloc(&params->attprojw, attprojw_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->attprojw, h_params + offset, attprojw_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += attprojw_size;
    printf("  Copied attprojw to device\n");
    
    cudaCheck(cudaMalloc(&params->fcw, fcw_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->fcw, h_params + offset, fcw_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += fcw_size;
    printf("  Copied fcw to device\n");
    
    cudaCheck(cudaMalloc(&params->fcprojw, fcprojw_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->fcprojw, h_params + offset, fcprojw_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += fcprojw_size;
    printf("  Copied fcprojw to device\n");
    
    cudaCheck(cudaMalloc(&params->ln1w, ln1w_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->ln1w, h_params + offset, ln1w_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += ln1w_size;
    printf("  Copied ln1w to device\n");
    
    cudaCheck(cudaMalloc(&params->ln1b, ln1b_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->ln1b, h_params + offset, ln1b_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += ln1b_size;
    printf("  Copied ln1b to device\n");
    
    cudaCheck(cudaMalloc(&params->ln2w, ln2w_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->ln2w, h_params + offset, ln2w_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += ln2w_size;
    printf("  Copied ln2w to device\n");
    
    cudaCheck(cudaMalloc(&params->ln2b, ln2b_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->ln2b, h_params + offset, ln2b_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += ln2b_size;
    printf("  Copied ln2b to device\n");
    
    cudaCheck(cudaMalloc(&params->qkvb, qkvb_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->qkvb, h_params + offset, qkvb_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += qkvb_size;
    printf("  Copied qkvb to device\n");
    
    cudaCheck(cudaMalloc(&params->fcb, fcb_size * sizeof(float)));
    cudaCheck(cudaMemcpy(params->fcb, h_params + offset, fcb_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += fcb_size;
    printf("  Copied fcb to device\n");
    
    free(h_params);
    
    printf("+----------------------------------+\n\n");
}

void free_scaled_params(ScaledParams* params) {
    cudaCheck(cudaFree(params->qkvw));
    cudaCheck(cudaFree(params->attprojw));
    cudaCheck(cudaFree(params->fcw));
    cudaCheck(cudaFree(params->fcprojw));
    cudaCheck(cudaFree(params->ln1w));
    cudaCheck(cudaFree(params->ln1b));
    cudaCheck(cudaFree(params->ln2w));
    cudaCheck(cudaFree(params->ln2b));
    cudaCheck(cudaFree(params->qkvb));
    cudaCheck(cudaFree(params->fcb));
}

// Structure to hold quantization results
typedef struct {
    float* q_factors;
    uint8_t* zero_points;
    uint32_t* quantized_packed;
    size_t num_q_factors;
    size_t num_zero_points;
    size_t num_packed;
} QuantizationResult;

// ============================================================================
// PROCESS ALL LAYERS FOR ONE WEIGHT TYPE (BATCHED - NO LOOP!)
// ============================================================================

QuantizationResult process_weight_type(
    float* d_weights_all_layers,  // Device pointer: all layers concatenated (L, N, C) - ALREADY SCALED!
    int L, int N, int C,
    const char* weight_name)
{
    printf("\n+---------------------------+\n");
    printf("| Processing %s\n", weight_name);
    printf("+---------------------------+\n");
    printf("| Layers: %d, N: %d, C: %d\n", L, N, C);
    printf("+---------------------------+\n");
    
    int num_groups_per_layer = C / 128;
    size_t total_groups = L * N * num_groups_per_layer;
    size_t packed_size = L * N * (C / 4);  // uint32_t packing: 4 uint8 per word
    
    // Note: Weights are already scaled by grid search - no additional scaling needed!
    
    // Allocate device buffers for quantization parameters
    float* d_q_factors;
    uint8_t* d_zero_points;
    cudaCheck(cudaMalloc(&d_q_factors, total_groups * sizeof(float)));
    cudaCheck(cudaMalloc(&d_zero_points, total_groups * sizeof(uint8_t)));
    
    // Allocate device buffer for packed quantized output
    uint32_t* d_quantized_packed;
    cudaCheck(cudaMalloc(&d_quantized_packed, packed_size * sizeof(uint32_t)));
    
    printf("  Computing quantization parameters for all %d layers...\n", L);
    
    // Step 1: Compute quantization parameters for ALL layers at once (batched)
    calc_quant_param(d_q_factors, d_zero_points, d_weights_all_layers, L, N, C);
    cudaCheck(cudaDeviceSynchronize());
    
    printf("  Quantizing weights for all %d layers...\n", L);
    
    // Step 2: Quantize ALL layers at once (batched)
    quantize_weights_uint32(d_quantized_packed, d_weights_all_layers,
                                    d_q_factors, d_zero_points, L, N, C);
    cudaCheck(cudaDeviceSynchronize());
    
    // Step 3: Copy results to host
    printf("  Copying results to host...\n");
    
    QuantizationResult result;
    result.num_q_factors = total_groups;
    result.num_zero_points = total_groups;
    result.num_packed = packed_size;
    
    result.q_factors = (float*)mallocCheck(total_groups * sizeof(float));
    result.zero_points = (uint8_t*)mallocCheck(total_groups * sizeof(uint8_t));
    result.quantized_packed = (uint32_t*)mallocCheck(packed_size * sizeof(uint32_t));
    
    cudaCheck(cudaMemcpy(result.q_factors, d_q_factors, total_groups * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(result.zero_points, d_zero_points, total_groups * sizeof(uint8_t), 
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(result.quantized_packed, d_quantized_packed, packed_size * sizeof(uint32_t), 
                         cudaMemcpyDeviceToHost));
    
    // Cleanup device memory
    cudaCheck(cudaFree(d_q_factors));
    cudaCheck(cudaFree(d_zero_points));
    cudaCheck(cudaFree(d_quantized_packed));
    
    printf("  Completed %s\n", weight_name);
    
    return result;
}

// ============================================================================
// SAVE AWQ CHECKPOINT
// ============================================================================

void save_awq_checkpoint(
    const char* filename,
    GPT2* model,
    ScaledParams* scaled_params,
    QuantizationResult* qkvw_result,
    QuantizationResult* attprojw_result,
    QuantizationResult* fcw_result,
    QuantizationResult* fcprojw_result)
{
    printf("\n+---------------------------+\n");
    printf("| Saving AWQ Checkpoint\n");
    printf("+---------------------------+\n");
    
    FILE* fp = fopenCheck(filename, "wb");
    
    // Extract model config
    int L = model->config.num_layers;
    int C = model->config.channels;
    int Vp = model->config.padded_vocab_size;
    int maxT = model->config.max_seq_len;
    int V = model->config.vocab_size;
    int NH = model->config.num_heads;
    
    // Write header (256 ints)
    int model_header[256] = {0};
    model_header[0] = 20250101;  // AWQ magic number
    model_header[1] = 1;          // version
    model_header[2] = maxT;       // max_seq_len
    model_header[3] = V;          // vocab_size
    model_header[4] = L;          // num_layers
    model_header[5] = NH;         // num_heads
    model_header[6] = C;          // channels
    model_header[7] = Vp;         // padded_vocab_size
    model_header[8] = 128;        // quant_group_size
    
    fwrite(model_header, sizeof(int), 256, fp);
    printf("  Wrote header\n");
    
    // Copy parameters from device to host
    float* params_host = (float*)mallocCheck(model->num_parameters * sizeof(float));
    cudaCheck(cudaMemcpy(params_host, model->params_memory, 
                         model->num_parameters * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Extract pointers using same logic as gpt2.cuh
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w (SCALED - use scaled_params)
    param_sizes[3] = L * C; // ln1b (SCALED - use scaled_params)
    param_sizes[4] = L * (3 * C) * C; // qkvw (skip - quantized)
    param_sizes[5] = L * (3 * C); // qkvb (SCALED - use scaled_params)
    param_sizes[6] = L * C * C; // attprojw (skip - quantized)
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w (SCALED - use scaled_params)
    param_sizes[9] = L * C; // ln2b (SCALED - use scaled_params)
    param_sizes[10] = L * (4 * C) * C; // fcw (skip - quantized)
    param_sizes[11] = L * (4 * C); // fcb (SCALED - use scaled_params)
    param_sizes[12] = L * C * (4 * C); // fcprojw (skip - quantized)
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
    
    // Calculate offsets
    size_t offsets[NUM_PARAMETER_TENSORS];
    offsets[0] = 0;
    for (int i = 1; i < NUM_PARAMETER_TENSORS; i++) {
        offsets[i] = offsets[i-1] + param_sizes[i-1];
    }
    
    // Allocate buffers for scaled parameters
    float* h_ln1w = (float*)mallocCheck(L * C * sizeof(float));
    float* h_ln1b = (float*)mallocCheck(L * C * sizeof(float));
    float* h_ln2w = (float*)mallocCheck(L * C * sizeof(float));
    float* h_ln2b = (float*)mallocCheck(L * C * sizeof(float));
    float* h_qkvb = (float*)mallocCheck(L * 3 * C * sizeof(float));
    float* h_fcb = (float*)mallocCheck(L * 4 * C * sizeof(float));
    
    // Copy scaled parameters from device
    cudaCheck(cudaMemcpy(h_ln1w, scaled_params->ln1w, L * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_ln1b, scaled_params->ln1b, L * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_ln2w, scaled_params->ln2w, L * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_ln2b, scaled_params->ln2b, L * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_qkvb, scaled_params->qkvb, L * 3 * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_fcb, scaled_params->fcb, L * 4 * C * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write non-quantized parameters in order expected by gpt2_awq.cuh
    // Order: wte, wpe, ln1w, ln1b, ln2w, ln2b, lnfw, lnfb, qkvb, attprojb, fcb, fcprojb
    
    fwrite(params_host + offsets[0], sizeof(float), param_sizes[0], fp); // wte
    fwrite(params_host + offsets[1], sizeof(float), param_sizes[1], fp); // wpe
    fwrite(h_ln1w, sizeof(float), param_sizes[2], fp); // ln1w (SCALED)
    fwrite(h_ln1b, sizeof(float), param_sizes[3], fp); // ln1b (SCALED)
    fwrite(h_ln2w, sizeof(float), param_sizes[8], fp); // ln2w (SCALED)
    fwrite(h_ln2b, sizeof(float), param_sizes[9], fp); // ln2b (SCALED)
    fwrite(params_host + offsets[14], sizeof(float), param_sizes[14], fp); // lnfw
    fwrite(params_host + offsets[15], sizeof(float), param_sizes[15], fp); // lnfb
    fwrite(h_qkvb, sizeof(float), param_sizes[5], fp); // qkvb (SCALED)
    fwrite(params_host + offsets[7], sizeof(float), param_sizes[7], fp); // attprojb
    fwrite(h_fcb, sizeof(float), param_sizes[11], fp); // fcb (SCALED)
    fwrite(params_host + offsets[13], sizeof(float), param_sizes[13], fp); // fcprojb
    
    // Free temporary buffers
    free(h_ln1w);
    free(h_ln1b);
    free(h_ln2w);
    free(h_ln2b);
    free(h_qkvb);
    free(h_fcb);
    
    printf("  Wrote non-quantized parameters (including scaled ln1w, ln1b, ln2w, ln2b, qkvb, fcb)\n");
    
    // Write dequantization factors (all floats)
    fwrite(qkvw_result->q_factors, sizeof(float), qkvw_result->num_q_factors, fp);
    fwrite(attprojw_result->q_factors, sizeof(float), attprojw_result->num_q_factors, fp);
    fwrite(fcw_result->q_factors, sizeof(float), fcw_result->num_q_factors, fp);
    fwrite(fcprojw_result->q_factors, sizeof(float), fcprojw_result->num_q_factors, fp);
    
    printf("  Wrote dequantization factors\n");
    
    // Write zero points (all uint8_t)
    fwrite(qkvw_result->zero_points, sizeof(uint8_t), qkvw_result->num_zero_points, fp);
    fwrite(attprojw_result->zero_points, sizeof(uint8_t), attprojw_result->num_zero_points, fp);
    fwrite(fcw_result->zero_points, sizeof(uint8_t), fcw_result->num_zero_points, fp);
    fwrite(fcprojw_result->zero_points, sizeof(uint8_t), fcprojw_result->num_zero_points, fp);
    
    printf("  Wrote zero points\n");
    
    // Write quantized weights (all uint32_t packed)
    fwrite(qkvw_result->quantized_packed, sizeof(uint32_t), qkvw_result->num_packed, fp);
    fwrite(attprojw_result->quantized_packed, sizeof(uint32_t), attprojw_result->num_packed, fp);
    fwrite(fcw_result->quantized_packed, sizeof(uint32_t), fcw_result->num_packed, fp);
    fwrite(fcprojw_result->quantized_packed, sizeof(uint32_t), fcprojw_result->num_packed, fp);
    
    printf("  Wrote quantized weights\n");
    
    fcloseCheck(fp);
    free(params_host);
    
    // Calculate and report file size
    size_t total_nq_params = param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] +
                             param_sizes[8] + param_sizes[9] + param_sizes[14] + param_sizes[15] +
                             param_sizes[5] + param_sizes[7] + param_sizes[11] + param_sizes[13];
    
    size_t total_dq_factors = qkvw_result->num_q_factors + attprojw_result->num_q_factors +
                              fcw_result->num_q_factors + fcprojw_result->num_q_factors;
    
    size_t total_zero_points = qkvw_result->num_zero_points + attprojw_result->num_zero_points +
                               fcw_result->num_zero_points + fcprojw_result->num_zero_points;
    
    size_t total_quantized = qkvw_result->num_packed + attprojw_result->num_packed +
                             fcw_result->num_packed + fcprojw_result->num_packed;
    
    size_t total_size = 256 * sizeof(int) +                        // header
                        total_nq_params * sizeof(float) +          // non-quantized params
                        total_dq_factors * sizeof(float) +         // dequant factors
                        total_zero_points * sizeof(uint8_t) +      // zero points
                        total_quantized * sizeof(uint32_t);        // quantized weights
    
    printf("\n+---------------------------+\n");
    printf("| AWQ Checkpoint Summary\n");
    printf("+---------------------------+\n");
    printf("  Non-quantized params: %zu (%zu MiB)\n", 
           total_nq_params, (total_nq_params * sizeof(float)) >> 20);
    printf("  Dequant factors: %zu (%zu MiB)\n", 
           total_dq_factors, (total_dq_factors * sizeof(float)) >> 20);
    printf("  Zero points: %zu (%zu KiB)\n", 
           total_zero_points, (total_zero_points * sizeof(uint8_t)) >> 10);
    printf("  Quantized weights: %zu uint32 (%zu MiB)\n", 
           total_quantized, (total_quantized * sizeof(uint32_t)) >> 20);
    printf("  Total file size: %zu MiB\n", total_size >> 20);
    printf("+---------------------------+\n");
    printf("  Saved to: %s\n", filename);
    printf("+---------------------------+\n\n");
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("\n");
    printf("=====================================================\n");
    printf("  AWQ Weight Quantization (4-bit, Packed)\n");
    printf("  Using Pre-Scaled Weights from Grid Search\n");
    printf("=====================================================\n\n");
    
    // Setup cuBLAS
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    
    // Load GPT2 model (only for non-quantized/non-scaled params like wte, wpe, lnfw, lnfb, attprojb, fcprojb)
    GPT2 model;
    gpt2_build_from_checkpoint(&model, checkpoint_file);
    
    int L = model.config.num_layers;
    int C = model.config.channels;
    
    printf("Model Configuration:\n");
    printf("  Layers (L): %d\n", L);
    printf("  Channels (C): %d\n", C);
    printf("  4*Channels: %d\n\n", 4 * C);
    
    // Load pre-scaled parameters from grid search
    ScaledParams scaled_params;
    load_scaled_params(scaled_params_filename, &scaled_params, L, C);
    
    printf("\n=====================================================\n");
    printf("  Starting Weight Quantization\n");
    printf("  (Using AWQ-optimized scaled weights)\n");
    printf("=====================================================\n");
    
    // Quantize each weight type using the pre-scaled weights
    // These weights already have optimal AWQ scaling applied
    
    // 1. QKV projection weights: (L, 3*C, C)
    QuantizationResult qkvw_result = process_weight_type(
        scaled_params.qkvw, L, 3*C, C, "QKV Weights");
    
    // 2. Attention projection weights: (L, C, C)
    QuantizationResult attprojw_result = process_weight_type(
        scaled_params.attprojw, L, C, C, "Attention Projection Weights");
    
    // 3. FC weights: (L, 4*C, C)
    QuantizationResult fcw_result = process_weight_type(
        scaled_params.fcw, L, 4*C, C, "FC Weights");
    
    // 4. FC projection weights: (L, C, 4*C)
    QuantizationResult fcprojw_result = process_weight_type(
        scaled_params.fcprojw, L, C, 4*C, "FC Projection Weights");
    
    printf("\n=====================================================\n");
    printf("  Weight Quantization Complete!\n");
    printf("=====================================================\n");
    
    // Save all results to single AWQ checkpoint file
    save_awq_checkpoint(output_checkpoint_file, &model, &scaled_params,
                       &qkvw_result, &attprojw_result, &fcw_result, &fcprojw_result);
    
    // Cleanup
    free(qkvw_result.q_factors);
    free(qkvw_result.zero_points);
    free(qkvw_result.quantized_packed);
    free(attprojw_result.q_factors);
    free(attprojw_result.zero_points);
    free(attprojw_result.quantized_packed);
    free(fcw_result.q_factors);
    free(fcw_result.zero_points);
    free(fcw_result.quantized_packed);
    free(fcprojw_result.q_factors);
    free(fcprojw_result.zero_points);
    free(fcprojw_result.quantized_packed);
    free_scaled_params(&scaled_params);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    
    printf("\n=====================================================\n");
    printf("  SUCCESS!\n");
    printf("  AWQ checkpoint saved to: %s\n", output_checkpoint_file);
    printf("=====================================================\n\n");
    
    return EXIT_SUCCESS;
}
